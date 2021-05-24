import torchvision.transforms as standard_transforms
from models.dbd_model import DBD
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from matplotlib import pyplot as plt
import torchvision.utils as vutils
from tools.running_logger import *
from tools.message_print import *
from tools.simple_timer import *
from tqdm import tqdm as tqdm
from tools.num_cal import *
from torch import optim
import torch
import numpy as np
import time

class Trainer(object):
    def __init__(self, output_path=None, config=None, cfg_data=None):
        self.output_path = output_path
        self.config = config
        self.cfg_data = cfg_data
        print("------------------use BLUR_map:", cfg_data.DATA_PATH)
        self.exp_name = time.strftime("%m-%d_%H-%M", time.localtime()) + '_' + config.exp_name
        self.exp_path = config.exp_path
        seed = config.seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        gpus = config.gpu_id
        if len(gpus) == 1:
             torch.cuda.set_device(gpus[0])
        net_name = config.net
        print('------------------use net: ', net_name)
        self.net = DBD(gpus, net_name, if_vector_loss=config.vector_loss,
                                use_my_loss=config.use_my_loss, gt_log_para=self.cfg_data.log_para).cuda()
        data_mode = config.data_mode
        print('------------------use dataset: ', data_mode)
        if config.optimizer == 'SGD':
            print('------------------use optimizer: SGD')
            self.optimizer = torch.optim.SGD(self.net.CCN.parameters(), config.lr,
                                             momentum=config.SGD.momentum, weight_decay=0)
        elif config.optimizer == 'Adam':
            print('------------------use optimizer: Adam')
            if self.config.net == 'SFAoatt':
                self.optimizer = optim.Adam(self.net.CCN.parameters(),
                                            lr=config.lr, weight_decay=1e-4)
            elif self.config.net == 'SFAoatthxq0515':
                self.optimizer = optim.Adam(filter(lambda p:p.requires_grad,self.net.CCN.parameters()),
                                            lr=config.lr, weight_decay=1e-4)
            else:
                self.optimizer = optim.Adam(self.net.CCN.parameters(),
                                            lr=config.lr, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=config.num_epoch_lr_decay,
                                gamma=config.lr_decay)
        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.train_record0 = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.train_record1 = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.train_record2 = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.train_record3 = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0

        self.dir_checkpoint='./exp_outputs/blur/blur-'

        if data_mode == 'BLUR':
            from datasets.BLUR.loading_data import loading_data
        self.train_loader, self.val_loader,self.restore_transform = loading_data(self.config)#
        if config.if_pretrain:
            print("fine tune from:", config.pretrained_path)
            pretrained_state = torch.load(config.pretrained_path)
            self.net.load_state_dict(pretrained_state['net'],strict=False)
        if config.if_resume:
            print("resume from:", config.resume_path)
            latest_state = torch.load(config.resume_path)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, data_mode)

    def forward(self):
        for epoch in range(self.config.MAX_EPOCH):
            if epoch < self.epoch:
                continue
            self.epoch = epoch
            print('This is epoch:---->', epoch + 1)
            self.timer['train time'].tic()
            self.train(epoch)
            self.timer['train time'].toc(average=False)
            if self.config.data_mode == 'BGI' and epoch % 10 == 0:
                pretrain_name = 'pretrained_epoch_' + str(epoch)
                to_saved_weight = self.net.state_dict()
                torch.save(to_saved_weight, os.path.join(self.exp_path, self.exp_name, pretrain_name + '.pth'))
            if epoch > self.config.lr_decay_start:
                print('decrease lr...')
                self.scheduler.step()
            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            if self.epoch % 20 == 0:
                state={'net':self.net.state_dict(),'optimizer':self.optimizer.state_dict(),'epoch':self.epoch}
                torch.save(state,self.dir_checkpoint+str(epoch)+'.pth')

    def train(self, epoch):
        self.net.train()
        for i, data in enumerate(tqdm(self.train_loader), 0):
            self.timer['iter time'].tic()
            img, gt_map, gt_adaptive = data
            self.optimizer.zero_grad()
            if self.config.net == 'SFAoatthxq0515':
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                pred_map = self.net(img, gt_map, epoch)

            else:
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                pred_map = self.net(img, gt_map,epoch)
            if self.config.net == 'SFAoatthxq0515':
                loss = self.net.loss
            else:
                loss = self.net.loss
            if self.config.vis_train_result:
                if self.epoch % 1 == 0 and i % 30 == 0:
                    if self.config.net == 'SFAoatt':
                        pred_map = pred_map.data.cpu().numpy()
                        gt_map = gt_map.data.cpu().numpy()
                        self.vis_results(exp_name=self.exp_name, epoch=self.epoch, writer=self.writer,
                                         restore=self.restore_transform, img=img, pred_map=pred_map,
                                         gt_map=gt_map, if_de=self.config.if_dedata,
                                         adptiver_gt_mask=gt_adaptive,
                                         pred=0, gt=0)

                    elif self.config.net == 'SFAoatthxq1101':
                        pred_map = pred_map.data.cpu().numpy()
                        gt_map = gt_map.data.cpu().numpy()
                        self.vis_results(exp_name=self.exp_name, epoch=self.epoch, writer=self.writer,
                                         restore=self.restore_transform, img=img, pred_map=pred_map,
                                         gt_map=gt_map, if_de=self.config.if_dedata,
                                         adptiver_gt_mask=gt_adaptive,
                                         pred=0, gt=0)

                    elif self.config.net == 'SFAoatthxq0515':
                        pred_map = pred_map.data.cpu().numpy()
                        gt_map = gt_map.data.cpu().numpy()
                        self.vis_results(exp_name=self.exp_name, epoch=self.epoch, writer=self.writer,
                                         restore=self.restore_transform, img=img, pred_map=pred_map,
                                         gt_map=gt_map, if_de=self.config.if_dedata,
                                         adptiver_gt_mask=gt_adaptive,
                                         pred=0, gt=0)

                    else:
                        confuse_map = confuse_map.data.cpu().numpy()
                        density_map0 = density_map0.data.cpu().numpy()
                        pred_map = pred_map.data.cpu().numpy()
                        gt_map = gt_map.data.cpu().numpy()
                        confuse_gt = confuse_gt.data.cpu().numpy()
                        confuse_useful = confuse_useful.data.cpu().numpy()
                        gt_adaptive = gt_adaptive.data.cpu().numpy()
                        gt_adaptive = (gt_adaptive > 0).astype(np.float32)

                        if self.config.train_batchsize == 1:
                            confuse_gt = confuse_gt.reshape(1, 1, confuse_gt.shape[0], confuse_gt.shape[1])
                            confuse_useful = confuse_useful.reshape(1, 1, confuse_useful.shape[0],
                                                                    confuse_useful.shape[1])
                        self.vis_results(exp_name=self.exp_name, epoch=self.epoch, writer=self.writer,
                                         restore=self.restore_transform, img=img, pred_map=pred_map,
                                         density_map0=density_map0,
                                         confuse_map=confuse_map, gt_map=gt_map, if_de=self.config.if_dedata,
                                         confuse_gt=confuse_gt, confuse_useful=confuse_useful,
                                         adptiver_gt_mask=gt_adaptive,
                                         pred=0, gt=0)

            if self.config.vector_loss:
                loss.backward(torch.ones_like(gt_map))
            else:
                loss.backward()

            self.optimizer.step()
            # print(i)
            if (i + 1) % self.config.PRINT_FREQ == 0:
                self.i_tb += 1
                if self.config.vector_loss:
                    self.writer.add_scalar('train_loss', loss.sum(), self.i_tb)
                else:
                    if self.config.net == 'VD2C':
                        self.writer.add_scalar('train_base_loss', loss0.item(), self.i_tb)
                        self.writer.add_scalar('train_confuse_loss', confuse_loss.item(), self.i_tb)
                        self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                    elif self.config.net == 'VD2C2':
                        self.writer.add_scalar('train_base_loss', loss0.item(), self.i_tb)
                        self.writer.add_scalar('train_confuse_loss', confuse_loss.item(), self.i_tb)
                        self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                    elif self.config.net == 'VD2C3':
                        self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                    else:
                        self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'] * 10000, self.i_tb)
                self.timer['iter time'].toc(average=False)
                if self.config.if_print:
                    if self.config.vector_loss:
                        print('[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' \
                              % (self.epoch + 1, i + 1, loss.sum(), self.optimizer.param_groups[0]['lr'] * 10000,
                                 self.timer['iter time'].diff))
                    else:
                        print('[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' \
                              % (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr'] * 10000,
                                 self.timer['iter time'].diff))
                    print('[cnt: gt: %.1f pred: %.2f]' % (
                        gt_map[0].sum().data / self.cfg_data.log_para, pred_map[0].sum().data / self.cfg_data.log_para))

    def validate(self):
        self.net.eval()
        losses = AverageMeter()
        confuse_losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        hxq_try1 = 0
        # data of all
        for vi, data in enumerate(tqdm(self.val_loader), 0):
            hxq_try1 += 1
            img, gt_map, gt_adaptive = data
            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                gt_adaptive = Variable(gt_adaptive).cuda()
                if self.config.net == 'NewV1':
                    pred_map, confuse_map = self.net.forward(
                        img, gt_map, self.epoch, gt_adaptive=gt_adaptive)
                    confuse_map = confuse_map.data.cpu().numpy()
                else:
                    pred_map = self.net.forward(img, gt_map, self.epoch)
                    confuse_map = None
                    density_map0 = None
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()
                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.log_para
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.log_para
                    if self.config.vector_loss:
                        losses.update(self.net.loss.sum())
                    else:
                        if self.config.net == 'VD2C3':
                            losses.update(self.net.loss.item())
                        else:
                            losses.update(self.net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                if self.epoch % 1 == 0 and vi % 20 == 0 and self.config.if_vis_results:
                    self.vis_results(exp_name=self.exp_name, epoch=self.epoch, writer=self.writer,
                                     restore=self.restore_transform, img=img, pred_map=pred_map,
                                     density_map0=density_map0,
                                     confuse_map=confuse_map, gt_map=gt_map, if_de=self.config.if_dedata,
                                     pred=pred_cnt, gt=gt_count, if_val=True)
        print(hxq_try1)
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg
        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        if self.config.net == 'SFAoatthxq0515':
            pass
        else:
            confuse_loss = confuse_losses.avg
            self.writer.add_scalar('val_confuse_loss', confuse_loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)
        #隔多少次保存参数
        if self.epoch % 5 == 0:
            self.train_record = self.update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb,
                                                  self.exp_path,
                                                  self.exp_name, [mae, mse, loss],
                                                  self.train_record, self.log_txt)
            print_summary(self.exp_name, [mae, mse, loss], self.train_record)


