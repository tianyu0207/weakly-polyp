import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
import torch.nn as nn


def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))



def ranking(scores, batch_size):
    loss = torch.tensor(0., requires_grad=True)
    for i in range(batch_size):
        # maxn = torch.max(scores[int(i*32):int((i+1)*32)] )
        # maxa = torch.max(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)])
        maxn =  torch.topk(scores[int(i*32):int((i+1)*32)], 3, largest=True)[0].mean()
        maxa = torch.topk(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)], 3, largest=True)[0].mean()
        tmp = F.relu(1.-maxa+maxn)
        loss = loss + tmp
        loss = loss + smooth(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)],8e-5)
        loss = loss + sparsity(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)], 8e-5)
    return loss / batch_size



class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        # print(q.shape)
        k = nn.functional.normalize(k, dim=1)
        # print(k.shape)
        
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        # print(neg.shape)
        # exit(1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['H_Abn'], 1), 
            torch.mean(contrast_pairs['E_Abn'], 1), 
            contrast_pairs['E_Nor']
        )

        # HA_refinement2 = self.NCE(
        #     torch.mean(contrast_pairs['H_Abn2'], 1), 
        #     torch.mean(contrast_pairs['E_Abn'], 1), 
        #     contrast_pairs['E_Abn']
        # )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['H_Nor_top_k'], 1), 
            torch.mean(contrast_pairs['E_Nor'], 1), 
            contrast_pairs['E_Abn']
        )

        loss = HA_refinement + HB_refinement 
        return loss
        


def train(nloader, aloader, model, batch_size, optimizer, device):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)
        
        x, cls_tokens, cls_prob,  scores, contrast_pairs, _ = model(input)  # b*32  x 2048
        
        scores = scores.view(batch_size * 32 * 2, -1)

        scores = scores.squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]
        
        label = torch.cat((nlabel, alabel), 0) 

        criterion = torch.nn.BCELoss()

        label = label.cuda()
        
        loss_cls = criterion(cls_prob.squeeze(1).squeeze(1), label)  # BCE loss in the video score space
        cost = ranking(scores, batch_size)  + loss_cls ##### DeepMIL CVPR 18
        snico_criterion = SniCoLoss()
        loss_snico = 0.25 * snico_criterion(contrast_pairs)
        cost += loss_snico
       
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


