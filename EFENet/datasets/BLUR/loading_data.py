import torchvision.transforms as standard_transforms
import datasets.data_preprocess as img_preprocess
from torch.utils.data import DataLoader
import torch
import random
from datasets.BLUR.base_cfg import cfg_data
from datasets.BLUR.BLUR import DATA_api

def get_min_size(batch):
    min_ht = cfg_data.TRAIN_SIZE[0]
    min_wd = cfg_data.TRAIN_SIZE[1]
    for i_sample in batch:
        _, ht, wd = i_sample.shape
        if ht < min_ht:
            min_ht = ht
        if wd < min_wd:
            min_wd = wd
    return min_ht, min_wd

def random_crop(img, den, dst_size):
    _, ts_hd, ts_wd = img.shape
    x1 = random.randint(0, ts_wd - dst_size[1]) // cfg_data.label_factor * cfg_data.label_factor
    y1 = random.randint(0, ts_hd - dst_size[0]) // cfg_data.label_factor * cfg_data.label_factor
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]
    label_x1 = x1 // cfg_data.label_factor
    label_y1 = y1 // cfg_data.label_factor
    label_x2 = x2 // cfg_data.label_factor
    label_y2 = y2 // cfg_data.label_factor
    return img[:, y1:y2, x1:x2], den[label_y1:label_y2, label_x1:label_x2]

def random_crop2(img, den, adaptive_den, dst_size):
    _, ts_hd, ts_wd = img.shape
    x1 = random.randint(0, ts_wd - dst_size[1]) // cfg_data.label_factor * cfg_data.label_factor
    y1 = random.randint(0, ts_hd - dst_size[0]) // cfg_data.label_factor * cfg_data.label_factor
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]
    label_x1 = x1 // cfg_data.label_factor
    label_y1 = y1 // cfg_data.label_factor
    label_x2 = x2 // cfg_data.label_factor
    label_y2 = y2 // cfg_data.label_factor
    return img[:, y1:y2, x1:x2], den[label_y1:label_y2, label_x1:label_x2], adaptive_den[label_y1:label_y2, label_x1:label_x2]

def share_memory(batch):
    out = None
    if False:
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out


def SHA_collate(batch):
    transposed = list(zip(*batch))
    imgs, dens = [transposed[0], transposed[1]]
    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):
        min_ht, min_wd = get_min_size(imgs)
        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop(imgs[i_sample], dens[i_sample], [min_ht, min_wd])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)
        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))
        return [cropped_imgs, cropped_dens]
    raise TypeError((error_msg.format(type(batch[0]))))

def SHA_collate2(batch):
    transposed = list(zip(*batch))
    imgs, dens, adaptive_dens = [transposed[0], transposed[1], transposed[2]]
    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor) and isinstance(adaptive_dens[0], torch.Tensor):
        min_ht, min_wd = get_min_size(imgs)
        cropped_imgs = []
        cropped_dens = []
        cropped_adaptive_dens = []
        for i_sample in range(len(batch)):
            _img, _den, _adaptive_den = random_crop2(imgs[i_sample], dens[i_sample], adaptive_dens[i_sample], [min_ht, min_wd])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)
            cropped_adaptive_dens.append(_adaptive_den)
        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))
        cropped_adaptive_dens = torch.stack(cropped_adaptive_dens, 0, out=share_memory(cropped_adaptive_dens))
        return [cropped_imgs, cropped_dens, cropped_adaptive_dens]
    raise TypeError((error_msg.format(type(batch[0]))))


def loading_data(config=None):
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.log_para
    factor = cfg_data.label_factor
    train_main_transform = img_preprocess.Compose([
        img_preprocess.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
    gt_transform = standard_transforms.Compose([
        img_preprocess.GTScaleDown(factor),
        img_preprocess.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
            img_preprocess.DeNormalize(*mean_std),
            standard_transforms.ToPILImage()
        ])
    train_set = DATA_api(cfg_data.DATA_PATH + '/train', 'train', main_transform=train_main_transform,
                         img_transform=img_transform, gt_transform=gt_transform, if_dedata=config.if_dedata)
    if config.train_batchsize == 1:
        train_loader = DataLoader(train_set, batch_size=config.train_batchsize, num_workers=8, collate_fn=SHA_collate2,
                                  shuffle=True,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_set, batch_size=config.train_batchsize, num_workers=8, collate_fn=SHA_collate2,
                                  shuffle=True,
                                  drop_last=True)
    val_set = DATA_api(cfg_data.DATA_PATH + '/test-xu', 'test', main_transform=None,
                       img_transform=img_transform, gt_transform=gt_transform, if_dedata=config.if_dedata)
    val_loader = DataLoader(val_set, batch_size=config.val_batchsize, num_workers=8, shuffle=False, drop_last=False)
    return train_loader, restore_transform , val_loader
