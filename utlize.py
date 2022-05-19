import numpy as np

from torch.utils.data import Dataset

import torch
from einops import rearrange
warnings.filterwarnings('ignore')

def getDataSet():
    '''
    load your own dataset with the shape of [n,wt,p],
    where n is the number of total cases sorted with chronological order
    wt is the number of wind turbine
    p is the number of considered parameters and the first parameter is wind speed
    :return:
    '''
    pass



class SCADADataset(Dataset):
    def __init__(self,windL ,predL ,wtnum):
        '''

        :param windL:
        :param predL:
        :param wtnum:
        '''
        self.data =getDataSet()
        self.predL =predL
        self.windL =windL
        self.wtnum =wtnum
        self.datashape = list(self.data.shape)

    def __len__(self):
        return self.data.shape[0] - self.windL - self.predL

    def __getitem__(self, idx):

        x = np.copy(self.data[idx:idx + self.windL, :, :])
        x = torch.from_numpy(x).float()

        y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, 0])
        y = torch.from_numpy(y).float()
        return x, y







def loadFromDict(dictM,key,default):
    try:
        return dictM[key],dictM
    except:
        try:
            dictM[key]=default
        except:
            dictM={}
            dictM[key] = default
        return default,dictM


def torchBetaF(a,b):
    upper=a.lgamma().exp()*b.lgamma().exp()
    lower=(a+b).lgamma().exp()+1e-10
    return upper/lower+1e-10

class BetaInverseCDF_Reparameter():
    def __init__(self,hyperParaModel):
        self.device = hyperParaModel['device']
        self.sampling_num = hyperParaModel['sampling_num']
    def __call__(self,alpha,beta):
        bs = alpha.shape[0]
        alpha = alpha.view(-1, 1)
        beta = beta.view(-1, 1)
        eps = torch.randn(bs, self.sampling_num).to(self.device)
        x1 = torch.exp(eps * torch.sqrt(1 / alpha + 1 / (alpha ** 2) / 2) + torch.log(alpha) - 1 / (2 * alpha))
        eps = torch.randn(bs, self.sampling_num).to(self.device)
        x2 = torch.exp(eps * torch.sqrt(1 / beta + 1 / (beta ** 2) / 2) + torch.log(beta) - 1 / (2 * beta))
        x3=x1/(x1+x2)
        return x3
def GMMloss(y,yPw,yPmu,yPlogvar,device,numberWT=1):
    if numberWT==1:
        n=yPw.shape[1]
        yPw = F.softmax(yPw, dim=-1)
        Psqvar = torch.exp(yPlogvar)
        # factor = 1 / math.sqrt(2 * math.pi)
        factor = 0.3989
        epsilon = torch.tensor(1e-5).to(device)
        loss = 0
        for nGmm in range(n):
            pmu = yPmu[:, nGmm].view(-1, 1)
            psqvar = Psqvar[:, nGmm].view(-1, 1) + epsilon
            pw = yPw[:, nGmm].view(-1, 1)
            temp = -(y - pmu) ** 2 / 2 / psqvar ** 2
            # prob=factor/psqvar*torch.exp(temp)
            # probwi=pw*prob
            loss += pw * factor / psqvar * torch.exp(temp)
        loss = torch.sum(-torch.log(loss + 1e-7))
        return loss
    else:
        y = rearrange(y, 'b s wt-> b wt s')
        n = yPw.shape[1]
        bs = yPw.shape[0]
        yPw = F.softmax(yPw, dim=-1)
        Psqvar = torch.exp(yPlogvar)
        # factor = 1 / math.sqrt(2 * math.pi)
        factor = 0.3989
        epsilon = torch.tensor(1e-5).to(device)
        loss = 0
        # loss += torch.sum(torch.log(yPw))
        # loss -= torch.sum(yPlogvar)
        for nGmm in range(n):
            xmu = (y - yPmu[:, :, nGmm].view(bs, -1, 1)) / (Psqvar[:, :, nGmm]**2).view(bs, -1, 1)
            up=torch.exp(-0.5*torch.norm(xmu,p=2,dim=-1)**2)
            down=torch.prod(Psqvar[:, :, nGmm].view(bs, -1, 1),dim=-1)+epsilon
            
            loss+=yPw[:, nGmm].view(-1, 1)*factor*(up/(down))
        loss = torch.sum(-torch.log(loss + 1e-7))
    return loss

