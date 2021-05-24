import torch
import network.pytorch_ssim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2

class wmyloss_cos(nn.Module):
    def __init__(self):
        super(wmyloss_cos, self).__init__()
    def forward(self, pred, gt,f1,f2,f3,f4,f5,f6,f7,f8):
        loss_fn = nn.MSELoss()
        loss1 = loss_fn(pred, gt)
        f=[f1,f2,f3,f4,f5,f6,f7,f8]
        model_num=8
        similarity_sum=0
        for i in range(model_num):
            f[i]=Variable(f[i], requires_grad = True)
            f[i] = f[i].view(f[i].size(0), -1)
        for i in range(0,model_num):
            for j in range(i+1,model_num):
                similarity_ij=torch.cosine_similarity(f[i],f[j],dim=1)
                similarity_sum+=similarity_ij
        similarity_sum=similarity_sum/28*3
        loss2=torch.mean(similarity_sum)
        loss=loss1+loss2*1.5
        print('loss：',loss)
        return loss

class wmyloss_dis(nn.Module):
    def __init__(self):
        super(wmyloss_dis, self).__init__()
        self.fc_pre = nn.Sequential(nn.Linear(256, 128, bias=False))
        self.fc = nn.Sequential(nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128), nn.BatchNorm1d(128))
        self.fc_pre.register_backward_hook(Adv_hook)

    def forward(self, pred, gt, f1, f2, f3):
        loss_fn = nn.MSELoss()
        loss1 = loss_fn(pred, gt)
        # print(f1.shape)
        f1 = Variable(f1, requires_grad=True)
        f2 = Variable(f2, requires_grad=True)
        f3 = Variable(f3, requires_grad=True)
        dis12 = self.cal_dis(f1, f2)
        dis13 = self.cal_dis(f1, f3)
        dis23 = self.cal_dis(f2, f3)
        loss2=dis12+dis13+dis23
        loss=loss1+loss2
        return loss


    def cal_dis(self,f1,f2):
        f1 = nn.functional.normalize(f1)
        f1 = self.fc_pre(f1)
        f1 = self.fc(f1)
        f1 = nn.functional.normalize(f1)

        f2 = nn.functional.normalize(f2)
        f2 = self.fc_pre(f2)
        f2 = self.fc(f2)
        f2 = nn.functional.normalize(f2)
        out = 0
        f = f1 - f2
        out = f.norm(dim=1, keepdim=True).mean()
        return out

def Adv_hook(module, grad_in, grad_out):
    return((grad_in[0] * (-1),grad_in[1]))


class hxqloss1(nn.Module):
    def __init__(self):
        super(hxqloss1, self).__init__()

    def forward(self, pred, gt):
        pred_final = torch.mean(pred,dim=-3)
        if not (pred_final.size() == gt.size()):
            print('Not the same size!!!')
            print('gt.shape = ' + str(gt.size()))
            print('pred_final.shape = ' + str(pred_final.size()))
            print('pred.shape = ' + str(pred.size()))
        loss = 0
        loss_channel = 32
        for i in range(loss_channel):
            if pred.shape[1] == loss_channel:
                G1 = pred[:,i,:,:]
            elif pred.shape[0] == loss_channel:
                G1 = pred[i,:,:]
            else:
                print('Pred.shape Error!!!')

            G1 = G1.squeeze()
            gt = gt.squeeze()
            loss += 0.5 * torch.mean(abs(G1 - gt)*abs(G1 - gt)) - 0.1 * torch.mean(abs(G1 - pred_final)*abs(G1 - pred_final))
        return loss



class hxqloss2(nn.Module):
    def __init__(self):
        super(hxqloss2, self).__init__()

    def forward(self, output32list, pred, gt):
        if not (pred.size() == gt.size()):
            print('Not the same size!!!')
            print('gt.shape = ' + str(gt.size()))
            print('pred.shape = ' + str(pred.size()))

        loss = 0
        loss_channel = 32
        final = torch.zeros(output32list[0].shape).cuda()
        for i in range(loss_channel):
            final += output32list[i]
        pred = final / 32

        for i in range(loss_channel):
            G1 = output32list[i]
            G1 = G1.squeeze()
            loss += 1 * torch.mean(abs(G1 - gt)*abs(G1 - gt)) - 0.05 * torch.mean(abs(G1 - pred)*abs(G1 - pred))
        return loss
