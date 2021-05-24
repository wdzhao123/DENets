import torchvision.transforms as standard_transforms
import datasets.data_preprocess as img_preprocess
from models.dbd_model import DBD
from matplotlib import pyplot as plt
from torch.autograd import Variable
from PIL import Image, ImageOps
from tools.num_cal import *
from tqdm import tqdm
import scipy.io as sio
from absl import flags
import pandas as pd
import numpy as np
import torch
import time
from time import *
import sys
import os
import cv2

# -------------parameters setting-----------
message = '在test集上测试,输出最终結果'
flags.DEFINE_list('gpu_id', [0], 'the GPU_ID')
flags.DEFINE_string('data_mode', 'BLUR', 'the dataset you run your model')
flags.DEFINE_string('net', 'SFAoatthxq0515', 'Number of train steps.')
flags.DEFINE_string('model_path',
                    "./exp_outputs/blur-590.pth",
                    'model path.')
flags.DEFINE_string('data_dir', "processed_data_size_15_4/test-dut", 'data_dir.')
flags.DEFINE_bool('if_de', False, 'if de dataset.')
gt_log_para = 100.
FLAGS = flags.FLAGS
FLAGS(sys.argv)
savepath='./result/result_dut/'
torch.cuda.set_device(0)
flags.mark_flag_as_required('gpu_id')
flags.mark_flag_as_required('data_mode')
flags.mark_flag_as_required('net')
flags.mark_flag_as_required('model_path')
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('if_de')
data_mode = FLAGS.data_mode
data_dir = FLAGS.data_dir
data_path = FLAGS.model_path
test_model = FLAGS.net
if_de = FLAGS.if_de
model_path = FLAGS.model_path

if data_mode == 'BLUR':
    from datasets.BLUR.base_cfg import cfg_data
mean_std = cfg_data.MEAN_STD
dataRoot = os.path.join('datasets', data_mode, data_dir)
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    img_preprocess.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()

def test(file_list, path):
    net = DBD(FLAGS.gpu_id, FLAGS.net)
    checkpoint=torch.load(path)
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()
    all_pics_cnt = 0
    for i in range(len(file_list)):
        all_pics_cnt += 1
        # imgname = dataRoot + '/img/' + str(i+1)+'.bmp'
        imgname = dataRoot + '/img/' + str(i+1)+'.jpg'
        print('imgname:',imgname)
        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map,list_f= net.test_forward(img)
            pred_map = torch.mean(pred_map, dim=-3)
        pred_map = pred_map.cpu().data.numpy().squeeze()
        pred_map = pred_map * (pred_map > 0)
        pred_map = pred_map*255
        pred_map= Image.fromarray(pred_map)
        pred_map=pred_map.resize((320,320))
        pred_map=pred_map.convert('L')
        pred_map.save(savepath+str(i+1)+'.bmp')
        plt.close()

def main():
    file_list = [filename for root, dirs, filename in os.walk(dataRoot + '/img/')]
    print('=================================================')
    print('----------net:', test_model)
    print('----------dataset:', data_mode)
    print('model_path:', model_path)
    begin_time = time()
    test(file_list[0], model_path)
    end_time = time()
    run_time = end_time - begin_time
    print('该程序运行时间：', run_time)


if __name__ == '__main__':
    main()
