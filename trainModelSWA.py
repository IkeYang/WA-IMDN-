#Author:ike yang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import datetime
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.autograd as autograd


import sys
sys.path.append("..")
sys.path.append("../../../../..")
from utlize import loadFromDict,SCADADataset,torchBetaF,GMMloss
from utlize import BetaInverseCDF_Reparameter as BetaInverseCDF

def compute_gradient_penalty( real_samples, fake_samples, labels,cuda=True):
    from modelMixMLP import MLPMixerReg as G
    from modelMixMLP import Discriminator as D
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    fake_samples = fake_samples.view(-1, 1)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    # F.gumbel_softmax
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def trainWAIMDN_GMM(deviceNum,nameInput,wtNum,wfnum,checkpoint=None,
                 hyperParaTrain=None,hyperParaModel=None,criterion=None):
    name='%s_%s_%s'%(nameInput,str(wfnum),str(wtNum))
    torch.cuda.set_device(deviceNum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    batch_size,hyperParaTrain=loadFromDict(hyperParaTrain,'batch_size',64)
    swa,hyperParaTrain=loadFromDict(hyperParaTrain,'swa',False)
    lrG,hyperParaTrain=loadFromDict(hyperParaTrain,'lrG',2e-3)#5 3
    lrD,hyperParaTrain=loadFromDict(hyperParaTrain,'lrD',2e-5)#5
    lambda_gp,hyperParaTrain=loadFromDict(hyperParaTrain,'lambda_gp',10)
    n_critic,hyperParaTrain=loadFromDict(hyperParaTrain,'n_critic',2)
    weight_decayG,hyperParaTrain=loadFromDict(hyperParaTrain,'weight_decayG',0)
    weight_decayD,hyperParaTrain=loadFromDict(hyperParaTrain,'weight_decayD',0)
    printOut,hyperParaTrain=loadFromDict(hyperParaTrain,'printOut',True)
    preTrain,hyperParaTrain=loadFromDict(hyperParaTrain,'preTrain',False)
    loadOpt,hyperParaTrain=loadFromDict(hyperParaTrain,'loadOpt',False)
    Maxepochs,hyperParaTrain=loadFromDict(hyperParaTrain,'Maxepochs',2000)
    tanhK, hyperParaTrain = loadFromDict(hyperParaTrain, 'tanhK', 4)
    windL,hyperParaTrain=loadFromDict(hyperParaTrain,'windL',50)
    predL,hyperParaTrain=loadFromDict(hyperParaTrain,'predL',6)
    varPun,hyperParaTrain=loadFromDict(hyperParaTrain,'varPun',0.05)
    evalName, hyperParaTrain = loadFromDict(hyperParaTrain, 'evalName', 'Val')
    EvalMSELoss, hyperParaTrain = loadFromDict(hyperParaTrain, 'EvalMSELoss',True)
    _, hyperParaTrain = loadFromDict(hyperParaTrain, 'wtNum',wtNum)
    modelName, hyperParaTrain = loadFromDict(hyperParaTrain, 'modelName','Mixer')
    n, hyperParaTrain = loadFromDict(hyperParaTrain, 'n', 5)
    prediction, hyperParaTrain = loadFromDict(hyperParaTrain, 'prediction', 'WindSpeed')
    _, hyperParaModel = loadFromDict(hyperParaModel, 'wtNum', wtNum)
    _, hyperParaModel = loadFromDict(hyperParaModel, 'kernelName', 'GMM')
    n,hyperParaModel = loadFromDict(hyperParaModel, 'n', n)
    sampling_num,hyperParaModel = loadFromDict(hyperParaModel, 'sampling_num', predL)
    gp_num,hyperParaModel = loadFromDict(hyperParaModel, 'gp_num', 60)
    out_dim ,hyperParaModel= loadFromDict(hyperParaModel, 'out_dim', 1)
    tanhK,hyperParaModel = loadFromDict(hyperParaModel, 'tanhK', tanhK)
    dropout,hyperParaModel = loadFromDict(hyperParaModel, 'dropout', 0.1)
    depth, hyperParaTrain = loadFromDict(hyperParaTrain, 'depth', 3)
    _,hyperParaModel = loadFromDict(hyperParaModel, 'depth', depth)
    _,hyperParaModel = loadFromDict(hyperParaModel, 'predL', predL)
    hiddenDim,hyperParaModel = loadFromDict(hyperParaModel, 'hiddenDim', None)



    scadaTrainDataset = SCADADataset( windL=windL, predL=predL, wtnum=wtNum)
    dataloader = torch.utils.data.DataLoader(scadaTrainDataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(0))


    inputShape, hyperParaModel = loadFromDict(hyperParaModel, 'inputShape', (windL,scadaTrainDataset.data.shape[1],scadaTrainDataset.data.shape[2]))  # inputshape T,wt,p

    embeddingD, hyperParaModel = loadFromDict(hyperParaModel, 'embeddingD', scadaTrainDataset.data.shape[1])




    generator = G(hyperParaModel).to(device)
    discriminator = D(hyperParaModel).to(device)
    paras_new = []
    for k, v in dict(generator.named_parameters()).items():
        if k == 'embedding.weight':
            paras_new += [{'params': [v], 'lr': lrG * 5}]
        else:
            paras_new += [{'params': [v], 'lr': lrG}]

    optimizer_G = torch.optim.Adam(paras_new, weight_decay=weight_decayG)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lrD, weight_decay=weight_decayD)
    if preTrain:
        generator.load_state_dict(checkpoint['G'])
        discriminator.load_state_dict(checkpoint['D'])
        if loadOpt:
            optimizer_G.load_state_dict(checkpoint['optimizerG'])
            optimizer_D.load_state_dict(checkpoint['optimizerD'])


    if criterion is None:
        def PunVarFun(fake_loss,w,mu,logstd,y=None):
            return -torch.mean(fake_loss) - varPun* torch.mean(torch.min(logstd, dim=-1)[0])
            #sometimes you can also punish mu and add additional auxiliary loss
            #ind = torch.argsort(w, dim=-1)[:, -2:]
            #wloss = torch.mean(torch.clamp(torch.var(w[ind], dim=-1)[0]-0.06,min=0))
            #mu = mu[ind]
            #muloss = torch.mean(torch.clamp(torch.sqrt((mu[:, 0]-mu[:, 1])**2),max=0.5))
            # muloss = torch.mean(torch.sqrt((mu[:, 0]-mu[:, 1])**2))
            # return -torch.mean(fake_loss) - varPun * torch.mean(torch.min(logstd, dim=-1)[0])\
            #  +wloss-muloss+GMMloss(y,w,mu,logstd,device)
        
            
        criterion=PunVarFun

    for epoch in range(Maxepochs):
        generator.train()
        discriminator.train()

        for i, (labels, y) in enumerate(dataloader):

            optimizer_D.zero_grad()
            bs = y.shape[0]
            labels = labels.to(device)

            y = y.to(device)

            w, mu, logstd = generator(labels)

            ws = F.softmax(w, dim=-1)
            w = F.softmax(w, dim=-1)
            std = torch.exp(logstd)
            w = w.view(-1, 1, n)
            w = w.repeat(1, sampling_num, 1)
            w = w.view(-1, n)

            eps = torch.randn(bs, sampling_num).to(device)  # bs label_dim
            sample = eps * (std[:, 0].view(bs, -1)) + mu[:, 0].view(bs, -1)
            fake = (ws[:, 0].view(-1, 1)) * sample
            fake_validity = discriminator(sample, labels)
            fake_loss = (w[:, 0].view(-1, 1)) * fake_validity
            for nGMM in range(n - 1):
                sample = eps * (std[:, nGMM + 1].view(bs, -1)) + mu[:, nGMM + 1].view(bs, -1)
                fake_validity = discriminator(sample, labels)
                fake_loss += (w[:, nGMM + 1].view(-1, 1)) * fake_validity
                fake += (ws[:, nGMM + 1].view(-1, 1)) * sample
            # Real
            real_validity = discriminator(y, labels)

            y = y.view(-1, 1)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, y.data, fake.data, labels.data)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_loss) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()


            optimizer_G.zero_grad()

            if i % n_critic == 0:
                w, mu, logstd = generator(labels)
                w = F.softmax(w, dim=-1)
                std = torch.exp(logstd)
                w = w.view(-1, 1, n)
                w = w.repeat(1, sampling_num, 1)
                w = w.view(-1, n)
                eps = torch.randn(bs, sampling_num).to(device)  # bs 360
                sample = eps * (std[:, 0].view(bs, -1)) + mu[:, 0].view(bs, -1)
                fake_validity = discriminator(sample, labels)
                fake_loss = (w[:, 0].view(-1, 1)) * fake_validity
                for nGMM in range(n - 1):
                    sample = eps * (std[:, nGMM + 1].view(bs, -1)) + mu[:, nGMM + 1].view(bs, -1)
                    fake_validity = discriminator(sample, labels)
                    fake_loss += (w[:, nGMM + 1].view(-1, 1)) * fake_validity
                g_loss =criterion(fake_loss,w,mu,logstd)
                g_loss.backward()
                optimizer_G.step()

def trainWAIMDN_Beta(deviceNum,nameInput,wtNum,wfnum,checkpoint=None,
                 hyperParaTrain=None,hyperParaModel=None,criterion=None,):
    from modelMixMLP import MLPMixerRegBeta as G
    # from modelMixMLP import MLPMixerReg as G
    from modelMixMLP import Discriminator as D
    name='%s_%s_%s'%(nameInput,str(wfnum),str(wtNum))
    torch.cuda.set_device(deviceNum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    prediction, hyperParaTrain = loadFromDict(hyperParaTrain, 'prediction', 'WindSpeed')
    batch_size,hyperParaTrain=loadFromDict(hyperParaTrain,'batch_size',64)
    swa,hyperParaTrain=loadFromDict(hyperParaTrain,'swa',False)
    lrG,hyperParaTrain=loadFromDict(hyperParaTrain,'lrG',2e-3)#5 3
    lrD,hyperParaTrain=loadFromDict(hyperParaTrain,'lrD',2e-5)#5
    lambda_gp,hyperParaTrain=loadFromDict(hyperParaTrain,'lambda_gp',10)
    n_critic,hyperParaTrain=loadFromDict(hyperParaTrain,'n_critic',2)
    weight_decayG,hyperParaTrain=loadFromDict(hyperParaTrain,'weight_decayG',0)
    weight_decayD,hyperParaTrain=loadFromDict(hyperParaTrain,'weight_decayD',0)
    printOut,hyperParaTrain=loadFromDict(hyperParaTrain,'printOut',True)
    preTrain,hyperParaTrain=loadFromDict(hyperParaTrain,'preTrain',False)
    loadOpt,hyperParaTrain=loadFromDict(hyperParaTrain,'loadOpt',False)
    Maxepochs,hyperParaTrain=loadFromDict(hyperParaTrain,'Maxepochs',2000)
    Maxepochs,hyperParaTrain=loadFromDict(hyperParaTrain,'Maxepochs',2000)
    windL,hyperParaTrain=loadFromDict(hyperParaTrain,'windL',50)
    predL,hyperParaTrain=loadFromDict(hyperParaTrain,'predL',6)
    varPun,hyperParaTrain=loadFromDict(hyperParaTrain,'varPun',0.05)
    evalName, hyperParaTrain = loadFromDict(hyperParaTrain, 'evalName', 'Val')
    BetaMLE, hyperParaTrain = loadFromDict(hyperParaTrain, 'BetaMLE',1)
    EvalMSELoss, hyperParaTrain = loadFromDict(hyperParaTrain, 'EvalMSELoss',True)
    _, hyperParaTrain = loadFromDict(hyperParaTrain, 'wtNum',wtNum)
    modelName, hyperParaTrain = loadFromDict(hyperParaTrain, 'modelName','Mixer')
    n, hyperParaTrain = loadFromDict(hyperParaTrain, 'n', 5)

    _, hyperParaModel = loadFromDict(hyperParaModel, 'wtNum', wtNum)
    n,hyperParaModel = loadFromDict(hyperParaModel, 'n', n)
    sampling_num,hyperParaModel = loadFromDict(hyperParaModel, 'sampling_num', predL)
    gp_num,hyperParaModel = loadFromDict(hyperParaModel, 'gp_num', 60)
    out_dim ,hyperParaModel= loadFromDict(hyperParaModel, 'out_dim', 1)
    tanhK,hyperParaModel = loadFromDict(hyperParaModel, 'tanhK', 4)
    _, hyperParaModel = loadFromDict(hyperParaModel, 'device', device)
    dropout,hyperParaModel = loadFromDict(hyperParaModel, 'dropout', 0.1)
    depth, hyperParaTrain = loadFromDict(hyperParaTrain, 'depth', 3)
    _,hyperParaModel = loadFromDict(hyperParaModel, 'depth', depth)
    _,hyperParaModel = loadFromDict(hyperParaModel, 'predL', predL)
    hiddenDim,hyperParaModel = loadFromDict(hyperParaModel, 'hiddenDim', None)



    scadaTrainDataset = SCADADataset(windL=windL, predL=predL, wtnum=wtNum,)
    dataloader = torch.utils.data.DataLoader(scadaTrainDataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(0))

    inputShape, hyperParaModel = loadFromDict(hyperParaModel, 'inputShape', (windL,scadaTrainDataset.data.shape[1],scadaTrainDataset.data.shape[2]))  # inputshape T,wt,p

    embeddingD, hyperParaModel = loadFromDict(hyperParaModel, 'embeddingD', scadaTrainDataset.data.shape[1])



    generator = G(hyperParaModel).to(device)
    discriminator = D(hyperParaModel).to(device)
    betaIcdf=BetaInverseCDF(hyperParaModel)
    paras_new = []
    for k, v in dict(generator.named_parameters()).items():
        if k == 'embedding.weight':
            paras_new += [{'params': [v], 'lr': lrG * 5}]
        else:
            paras_new += [{'params': [v], 'lr': lrG}]


    optimizer_G = torch.optim.Adam(paras_new, weight_decay=weight_decayG)

    def weights_init(m):
        invert_op = getattr(m, "weight", None)
        if callable(invert_op):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    discriminator.apply(weights_init)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lrD, weight_decay=weight_decayD)

    if preTrain:
        generator.load_state_dict(checkpoint['G'])
        discriminator.load_state_dict(checkpoint['D'])
        if loadOpt:
            optimizer_G.load_state_dict(checkpoint['optimizerG'])
            optimizer_D.load_state_dict(checkpoint['optimizerD'])


    if criterion is None:
        def PunVarFun(fake_loss,w,alpha,beta,y,BetaMLE):
            loss = 0
            for nGmm in range(n):
                palpha = alpha[:, nGmm].view(-1, 1)
                pbeta = beta[:, nGmm].view(-1, 1)
                pw = w[:, nGmm].view(-1, 1)
                B = torchBetaF(palpha, pbeta) + 1e-5
                lookB = B.cpu().detach().numpy()
                if np.sum(np.isnan(lookB)) > 0:
                    print('nan B')
                    restart = True
                    return None
                temp = (y + 1e-5) ** (palpha - 1) * (1 - y + 1e-5) ** (pbeta - 1)
                loss += pw / B * temp
            loss=torch.sum(-torch.log(loss + 1e-5))
            # if BetaMLE<0.02:
            #     return -torch.mean(fake_loss)
            return -torch.mean(fake_loss)+BetaMLE*loss
        criterion=PunVarFun





    for epoch in range(Maxepochs):



        generator.train()
        discriminator.train()
        BetaMLE=1/(epoch+1)

        for i, (labels, y) in enumerate(dataloader):

            optimizer_D.zero_grad()
            bs = y.shape[0]
            labels = labels.to(device)


            y = y.to(device)  # y=bs,(w,mu,logvar)

            w, alpha, Beta = generator(labels)# 3* bs ,n


            ws = F.softmax(w, dim=-1)

            w = F.softmax(w, dim=-1)
            w = w.view(-1, 1, n)
            w = w.repeat(1, sampling_num, 1)
            w = w.view(-1, n)


            #n w adding

            sample =betaIcdf(alpha[:,0],Beta[:,0])
            fake = (ws[:, 0].view(-1, 1)) * sample
            fake_validity = discriminator(sample, labels)
            fake_loss = (w[:, 0].view(-1, 1)) * fake_validity
            for nGMM in range(n - 1):
                sample = betaIcdf(alpha[:, nGMM + 1], Beta[:, nGMM + 1] )
                fake_validity = discriminator(sample, labels)
                fake_loss +=(w[:, nGMM + 1].view(-1, 1))  * fake_validity
                fake += (ws[:, nGMM + 1].view(-1, 1)) * sample
            # Real
            real_validity = discriminator(y, labels)

            y = y.view(-1, 1)

            gradient_penalty = compute_gradient_penalty(discriminator, y.data, fake[:,:predL].data, labels.data)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_loss) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            y=y.view(bs,-1)

            optimizer_G.zero_grad()

            if i % n_critic == 0:


                w, alpha, Beta= generator(labels)

                ws = F.softmax(w, dim=-1)
                w = F.softmax(w, dim=-1)

                w = w.view(-1, 1, n)
                w = w.repeat(1, sampling_num, 1)
                w = w.view(-1, n)

                sample = betaIcdf(alpha[:, 0], Beta[:, 0])
                fake_validity = discriminator(sample, labels)
                fake_loss = (w[:, 0].view(-1, 1)) * fake_validity
                for nGMM in range(n - 1):
                    sample = betaIcdf(alpha[:, nGMM + 1], Beta[:, nGMM + 1])
                    fake_validity = discriminator(sample, labels)
                    fake_loss +=(w[:, nGMM + 1].view(-1, 1))  * fake_validity
                g_loss =criterion(fake_loss,ws,alpha,Beta,y,BetaMLE)

                g_loss.backward()
                optimizer_G.step()

if __name__=='__main__':

    hyperParaTrain={'preTrain':False,'printOut':False,'n':3,'Maxepochs':1000 ,
                    'varPun': 0.01, 'tanhK': 10}
    trainWAIMDN_GMM(nameInput='TestGMM',wtNum=0,wfnum=4,deviceNum=0,hyperParaTrain=hyperParaTrain,hyperParaModel=None,criterion=None)
    trainWAIMDN_Beta(nameInput='TestBMM',wtNum=0,wfnum=4,deviceNum=0,hyperParaTrain=hyperParaTrain,hyperParaModel=None,criterion=None)





































































