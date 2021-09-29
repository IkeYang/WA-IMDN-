import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from torch.nn.parameter import Parameter



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, num_channels, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange('b p c -> b c p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b c p-> b p c')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(num_channels),
            FeedForward(num_channels, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x




from einops import rearrange

class Embed(nn.Module):
    def __init__(self, p,embedingDim,tuo=1,wt=33,wl=50,numberWT=1):
        super().__init__()
        self.relu=nn.ReLU()
        self.sftm=nn.Softmax(dim=-1)
        self.tuo=tuo
        self.thres=1/p/5/wt
        self.weight = Parameter(
            torch.ones(int(p*wt*wl)))

        self.FC=nn.Linear(p*wt,wt*numberWT)
        self.wl=wl
        self.wt=wt
        self.p=p
    def forward(self, x):
        bs,c,wt,p=x.shape[0],self.wl, self.wt,self.p
        x=x.view(bs,-1)

        w = self.sftm(self.weight/self.tuo)
        # w=torch.where(w>self.thres,torch.ones_like(w),torch.zeros_like(w))
        # w=self.sftm(self.relu(w-self.thres)+1e3)
        x=x * w
        x = x.view(bs, c, -1)
        x =self.FC(x).squeeze(dim=-1)
        x=rearrange(x,'b c p-> b p c')### p=wt*numberWT,c=wl
        return x

class MLPMixerReg(nn.Module):

    def __init__(self,hyperParaModel):
        wl,wtTotalNum,p=hyperParaModel['inputShape']
        hiddenDim=hyperParaModel['hiddenDim']
        embedingDim=hyperParaModel['embeddingD']#wt
        n=hyperParaModel['n']
        # self.modelName=hyperParaModel['modelName']
        depth=hyperParaModel['depth']
        self.tanh=hyperParaModel['tanhK']
        self.wtNum=hyperParaModel['wtNum']
        if type(self.wtNum) is list:
            self.numberWT=len(self.wtNum)
        else:
            self.numberWT=1
        super().__init__()
        self.n =n

        self.embedding = Embed(p,embedingDim,wt=wtTotalNum,wl=wl,numberWT=self.numberWT)
        self.Tanh = nn.Tanh()
        self.mixer_blocks = nn.ModuleList([])
        if  hiddenDim is None:
            hiddenDim=wl
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(wl, embedingDim*self.numberWT, wtTotalNum*self.numberWT, hiddenDim))

        self.layer_norm = nn.LayerNorm(wl)

        self.mlp_head = nn.Sequential(
            nn.Linear(wl, wl*2*self.numberWT),
            nn.ReLU(),
            nn.Linear(wl*2*self.numberWT, wl*2*self.numberWT),
            nn.ReLU(),
            nn.Linear(wl*2*self.numberWT, self.n+self.n*self.numberWT+self.n*self.numberWT*self.numberWT),
        )

    def forward(self, x):

        #x, bs,wl,wt,p

        x = self.embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        x=self.mlp_head(x)

        if self.numberWT==1:
            x=x.view(-1,self.numberWT, self.n, 3)
            return x[:, 0,:, 0], x[:, 0,:, 1], self.Tanh(x[:, 0,:, 2]) * self.tanh # w,mu logsigma
        else:

            sigema=x[:, self.n+self.n*self.numberWT:].view(-1,self.numberWT*self.numberWT, self.n)
            sigema = rearrange(sigema, 'b wt n-> b n wt') .view(-1,self.n,self.numberWT,self.numberWT)

            diagSigema = torch.diag_embed(torch.diagonal(sigema, dim1=-1, dim2=-2), dim1=-1, dim2=-2)
            sigema= sigema-diagSigema+torch.exp(diagSigema)
            sigema = torch.tril(sigema)
            # sigemaT = torch.transpose(sigema,dim0=-2,dim1=-1)
            # sigema=torch.matmul(sigema,sigemaT)

            return x[:, :self.n], x[:, self.n:(self.n+self.n*self.numberWT)].view(-1, self.n,self.numberWT), \
                   self.Tanh(sigema) * self.tanh**0.5

class MLPMixerRegBeta(nn.Module):

    def __init__(self,hyperParaModel):
        wl,wtTotalNum,p=hyperParaModel['inputShape']
        hiddenDim=hyperParaModel['hiddenDim']
        embedingDim=hyperParaModel['embeddingD']
        n=hyperParaModel['n']
        depth=hyperParaModel['depth']
        self.tanh=hyperParaModel['tanhK']
        super().__init__()




        self.n =n

        self.embedding = Embed(p,embedingDim,wt=wtTotalNum,wl=wl)
        self.Tanh = nn.Tanh()

        self.mixer_blocks = nn.ModuleList([])
        if  hiddenDim is None:
            hiddenDim=wl
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(wl, embedingDim, wtTotalNum, hiddenDim))

        self.layer_norm = nn.LayerNorm(wl)
        self.relu = nn.ReLU()
        self.mlp_head = nn.Sequential(
            nn.Linear(wl, wl*2),
            nn.ReLU(),
            nn.Linear(wl*2, wl*2),
            nn.ReLU(),
            nn.Linear(wl*2, self.n*3),
        )

    def forward(self, x):

        #x, bs,wl,wt,p

        x = self.embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        x=self.mlp_head(x).view(-1, self.n, 3)


        # return x[:, :, 0], self.relu(x[:, :,1])+1, self.relu(x[:, :,2])+1 # w,mu logsigma

        w, alpha, Beta  =x[:, :, 0], self.relu(x[:, :,1])+1, self.relu(x[:, :,2]) +1# w,mu logsigma

        return  w, alpha, Beta

class Discriminator(nn.Module):
    def __init__(self,hyperParaModel ):

        super(Discriminator, self).__init__()
        sampling_num = hyperParaModel['sampling_num']
        embeddingD = hyperParaModel['embeddingD']
        self.wtNum = hyperParaModel['wtNum']
        if type(self.wtNum) is list:
            self.numberWT = len(self.wtNum)
        else:
            self.numberWT = 1
        out_dim = 1
        wl, wtTotalNum, p = hyperParaModel['inputShape']
        n = hyperParaModel['n']

        super().__init__()
        self.n = n

        self.embedding = Embed(p, embeddingD)

        self.sftmW = nn.Softmax(dim=0)
        self.sampling_num=sampling_num
        self.embeddingD=embeddingD
        self.label_dim=wl*embeddingD
        self.label_dim=wl
        # Copied from cgan.py

        self.model = nn.Sequential(
            nn.Linear(self.label_dim*self.numberWT + out_dim*self.numberWT, self.label_dim*2*self.numberWT),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.label_dim*2*self.numberWT, self.label_dim*self.numberWT),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.label_dim*self.numberWT, 1),
        )

    def forward(self, sample, labels):
        #sample bs,self.numberWT
        #labels bs,wl,self.numberWT,p
        if self.numberWT == 1:
            bs=labels.shape[0]
            sample=sample.view(bs,-1)
            self.sampling_num=sample.shape[1]
            # labels=self.embedding(labels)
            labels=labels[:,:,self.wtNum,0]
            labels = labels.reshape(bs, 1, -1)
            labels = labels.repeat(1, self.sampling_num, 1)
            labels = labels.view(bs* self.sampling_num,-1)
            sample=sample.view(-1,1)

            d_in = torch.cat((labels, sample),1)
            validity = self.model(d_in)
            return validity
        else:
            # sample = rearrange(sample, 'b wt s-> b s wt')
            label=labels[:,:,:,0]
            bs = label.shape[0]
            # sample = sample.view(bs, -1)
            self.sampling_num = sample.shape[1]
            # sample = sample.view(bs, -1)
            # labels=self.embedding(labels)

            label = label.reshape(bs, 1, -1)
            label = label.repeat(1, self.sampling_num, 1)
            label = label.view(bs * self.sampling_num, -1)
            sample = sample.reshape(bs * self.sampling_num, -1)

            d_in = torch.cat((label, sample), 1)
            validity = self.model(d_in)
            return validity




