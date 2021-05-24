from easydict import EasyDict as edict

__C_BLUR = edict()
cfg_data = __C_BLUR
__C_BLUR.TRAIN_SIZE = (320, 320)
__C_BLUR.DATA_PATH = 'datasets/BLUR/processed_data_size_15_4'
__C_BLUR.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
__C_BLUR.log_para = 1
__C_BLUR.label_factor = 2
