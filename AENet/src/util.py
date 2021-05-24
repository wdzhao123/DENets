import numpy as np
import ipdb
import cv2

# BGR mean pixel value
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

"""Padding image and segmentation ground truth to (640, 640)"""
def prep_im_for_blob(im_name, max_size=(320,320)):
    im = cv2.imread(im_name)    # OpenCV color map default BGR
    #print im
    # max_size = (im.shape[0], im.shape[1])
    im_blob = np.zeros((max_size[0], max_size[1], 3))
    im_blob = cv2.resize(im, max_size, interpolation=cv2.INTER_NEAREST)
    im_blob = im_blob - MEAN_PIXEL

    row, col, _ = im.shape
    #im_blob = np.zeros((max_size[0], max_size[1], 3))
    #im_blob[0:row,0:col,:] = im

    return {'im_blob':im_blob}

"""Minus mean pixel value"""
def prep_im(im_name):
    im = cv2.imread(im_name)    # OpenCV color map default BGR
    im = np.array([im - MEAN_PIXEL])
    return im

"""For multi-processing dataloader"""
def prep_run_wrapper(args):
    return prep_im_for_blob(*args)

"""Resize input image to (256, 256)"""
def prep_small_im_for_blob(im_name, seg_name, max_size=(320,320)):
    im = cv2.imread(im_name)    # OpenCV color map default BGR
    #seg = cv2.imread(seg_name)[:,:,::-1]
    # max_size = (im.shape[0], im.shape[1])

    seg = cv2.imread(seg_name, cv2.COLOR_BGR2GRAY)

    row, col, _ = im.shape
    im_blob = np.zeros((max_size[0], max_size[1], 3))
    im_blob = cv2.resize(im, max_size, interpolation=cv2.INTER_NEAREST)
    im_blob = im_blob - MEAN_PIXEL

    seg_blob = np.zeros((max_size[0], max_size[1]))
    seg_blob = cv2.resize(seg, max_size, interpolation=cv2.INTER_NEAREST)
    seg_blob /= 255
    seg_blob = seg_blob[:, :, np.newaxis]
    #print seg_blob
    return {'im_blob':im_blob, 'seg_blob':seg_blob,'original_size':(row,col)}

"""For multi-processing dataloader"""
def prep_small_run_wrapper(args):
    return prep_small_im_for_blob(*args)


"""Create color mappings, check VOClabelcolormap.m for reference"""
def colormap(N=256):
    # Create double side mappings
    gray_to_rgb = {}
    rgb_to_gray = {}

    for i in range(N):
        temp = i
        r = 0
        g = 0
        b = 0
        for j in range(8):
            r = r | ((temp & 1) << (7-j))
            g = g | (((temp >> 1) & 1) << (7-j))
            b = b | (((temp >> 2) & 1) << (7-j))
            temp = temp >> 3
        gray_to_rgb[i] = (r,g,b)

    for key, val in gray_to_rgb.iteritems():
        rgb_to_gray[val] = key

    return gray_to_rgb, rgb_to_gray

"""Get original size"""
def get_original_size(mask, max_size=(640,640)):
    row = None
    col = None
    for i in range(max_size[0]-1, -1, -1):
        if mask[i,0,0] == 1:
            row = i + 1
            break

    for i in range(max_size[1]-1, -1, -1):
        if mask[0,i,0] == 1:
            col = i + 1
            break

    if row is None or col is None:
        ipdb.set_trace()
    return row, col

"""Transform gray scale segmentation result to rgb format"""
def seg_gray_to_rgb(seg, gray_to_rgb):
    row, col = seg.shape
    rgb = np.zeros((row, col, 3))

    for i in range(row):
        for j in range(col):
            r, g, b = gray_to_rgb[seg[i, j]]
            rgb[i, j, 0] = r
            rgb[i, j, 1] = g
            rgb[i, j, 2] = b

    return rgb


"""
Helper functions for bilinear upsampling
credit: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
"""
def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in xrange(number_of_classes): 
        weights[:, :, i, i] = upsample_kernel

    return weights

