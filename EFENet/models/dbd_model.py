import time
import torch.nn as nn
from network.loss import *

class DBD(nn.Module):
    def __init__(self, gpus, model_name, if_vector_loss=False, use_my_loss=False, gt_log_para=100):
        super(DBD, self).__init__()
        self.net_name = model_name
        self.gt_log_para = gt_log_para
        if model_name == 'VGG':
            from models.backbones.VGG import VGG as net
        elif model_name == 'SFAoatt':
            from models.backbones.SFAoatt import SFAoatt as net
        elif model_name == 'SFAoatthxq1101':
            from models.backbones.SFAoatthxq1101 import SFAoatthxq1101 as net
        elif model_name == 'SFAoatthxq0515':
            from models.backbones.SFAoatthxq0515 import SFAoatthxq0515 as net
        self.CCN = net()
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        if model_name == 'PSP':
            empty_dict = self.CCN.state_dict()
            checkpoint_para = torch.load(
                'pretrained_models/epoch_58_iter_192_loss_0.20348_acc_0.95038_acc-cls_0.80996_mean-iu_0.71921_fwavacc_0.90907_lr_0.0019853885.pth')
            load_dict = {k: v for k, v in checkpoint_para.items() if
                         k != 'module.final.4.weight' and k != 'module.final.4.bias' and k != 'module.aux_logits.weight' and k != 'module.aux_logits.bias'}
            empty_dict.update(load_dict)
            self.CCN.load_state_dict(empty_dict)
        if if_vector_loss:
            self.loss_mse_fn = nn.MSELoss(reduction='none').cuda()
        else:
            if model_name == 'CAN' or 'CANd8':
                # self.loss_mse_fn = nn.MSELoss(size_average=False).cuda()
                self.loss_mse_fn = nn.MSELoss().cuda()
            else:
                self.loss_mse_fn = nn.MSELoss().cuda()
        self.use_my_loss = use_my_loss
        if self.use_my_loss == 'hxqloss1':
            self.my_loss_mse_fn = hxqloss1().cuda()
        elif self.use_my_loss == 'hxqloss2':
            self.my_loss_mse_fn = hxqloss2().cuda()
        elif self.use_my_loss == 'hxqloss0515':
            self.my_loss_mse_fn = hxqloss0515().cuda()
        else:
            print('Loss Error!!!')

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_map, epoch, gt_adaptive=None):
        if self.net_name == 'SFAoatthxq0515':
            density_map, outlist = self.CCN(img)
            self.loss_mse = self.build_loss(outputlist=outlist,density_map=density_map.squeeze(), gt_data=gt_map.squeeze())#
            density_map = torch.mean(density_map,dim=-3)
            return density_map

        elif self.net_name == 'SFAoatt_wmy':
            density_map, outlist = self.CCN(img)
            self.loss_mse = self.build_loss(outputlist=outlist, density_map=density_map.squeeze(), gt_data=gt_map.squeeze())
            density_map = torch.mean(density_map,dim=-3)
            return density_map

        else:
            density_map = self.CCN(img)
            self.loss_mse = self.build_loss(density_map=density_map.squeeze(), gt_data=gt_map.squeeze())
            return density_map

    def build_loss(self, outputlist=None,density_map=None, confuse_map=None, fusion_map=None, gt_data=None, epoch=None,
                   gt_adaptive=None, map_middle_out_1=None, confuse_middle_out_1=None, map_middle_out_0=None,
                   confuse_middle_out_0=None):
        tic = time.time()

        if self.use_my_loss == 'hxqloss1':
            loss_mse = self.my_loss_mse_fn(density_map, gt_data)
        elif self.use_my_loss == 'hxqloss2':
            loss_mse = self.my_loss_mse_fn(outputlist,density_map, gt_data)
        elif self.use_my_loss == 'hxqloss0515':
            loss_mse = self.my_loss_mse_fn(density_map, gt_data, outputlist)#
        else:
            loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def test_forward(self, img):
        if self.net_name == 'SFAoatthxq0515':
            density_map, list_f = self.CCN(img) #
        else:
            density_map = self.CCN(img)
        return density_map,list_f
