import cv2
import numpy as np
import os
def train_data(iter):
    # image_path = '/home/hxq/PycharmProjects/tf0521/result0625_32_2a/32channels_denoise_dataset'
    # figs= os.listdir(image_path) # 该函数返回指定的文件夹包含的文件或文件夹的名字的列表。
    # num = len(figs)
    # for fig_path in sorted(os.listdir(image_path)):
    #     a = fig_path.split("fig")[1]
    #     j = int(a)-1
    #     path = image_path+'/'+fig_path+'/'
    #     img0 = cv2.imread(path+'channel1.bmp',0)
    #     img0 = np.expand_dims(img0,axis=-1)
    #     # train.npy
    #     for i in range(1,32):
    #         img = cv2.imread(path+'channel'+str(i+1)+'.bmp',0)
    #         img = np.expand_dims(img, axis=-1)
    #         # OpenCV color map default BGR
    #         img0 = np.concatenate((img0,img),axis=2)
    #     img0 = img0 / 256.0
    # return img0
    train_path = '/home/hxq/PycharmProjects/tf0521/result0625_32_2a/32channels_denoise_dataset/fig'+str(iter)+'/'
    for i in range(1,32):
        img = cv2.imread(train_path+'channel'+str(i+1)+'.bmp',0)
        img = np.expand_dims(img, axis=-1)
        img0 = np.concatenate((img0,img),axis=2)
    img0 = img0 / 256.0
    return img0
def label_data(iter):
    label_path = '/home/hxq/caffe_test/Label1/'
    seg = cv2.imread(label_path+str(iter)+'.bmp')
    seg = cv2.resize(seg,(320,320,3),interpolation=cv2.INTER_CUBIC)
    seg = seg / 256.0
    return seg





















#普通的train和label, 制作数据集
# import cv2
# import numpy as np
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
#
# # image_path = '/home/hxq/caffe_test/Train1'
# # image_path = '/home/hxq/caffe_test/Label1'
# image_path = '/home/hxq/caffe_test/test-dataset-320/blur100source'
# # image_path = '/home/hxq/caffe_test/test-dataset-320/blur100gt'
#
# figures = os.listdir(image_path) # 该函数返回指定的文件夹包含的文件或文件夹的名字的列表。
#
# num = len(figures)
# MEAN_PIXEL = np.array([103.939, 116.779, 123.68])
#
# # img_shape = np.zeros((num,320,320,1))
# img_shape = np.zeros((num,320,320,3))
#
# # for i,image in enumerate(figures): # enumerate 函数用于遍历序列中的元素以及它们的下标，
#                                     # 但是遍历文件顺序时不一定按数字
#
# for image in sorted(os.listdir(image_path)):
#     a = image.split(".")[0]
#     i = int(a)-1
#     print(i)
#
#     # train.npy
#     img = cv2.imread(os.path.join(image_path,image))
#     img = cv2.resize(img,(320,320),interpolation=cv2.INTER_CUBIC)
#     img = img - MEAN_PIXEL
#     img_shape[i] = img
#
#     # # label.npy
#     # img = cv2.imread(os.path.join(image_path,image),0) # 0指灰度图像读取，os.path.join指路径拼接
#     # img = cv2.resize(img,(320,320),interpolation=cv2.INTER_CUBIC)
#     # img = np.expand_dims(img,axis=-1)
#     # img = img / 256.0
#     # img_shape[i] = img
#
# # print(img_shape)
#
# # np.save('/home/hxq/PycharmProjects/tf-try/venv/train.npy',img_shape)
# # np.save('/home/hxq/PycharmProjects/tf-try/venv/label.npy',img_shape)
# np.save('/home/hxq/PycharmProjects/tf-try/venv/train100.npy',img_shape)
# # np.save('/home/hxq/PycharmProjects/tf-try/venv/label100.npy',img_shape)
#
#
# print (img_shape.shape)



























# import cv2
# import numpy as np
#
#
# class prepocessing(object):
#
#     def __init__(self):
#         self.path = '/home/hxq/caffe_test/test-dataset-320/blur100gt/'
#         self.num = 100
#         self.height = 320
#         self.weight = 320
#         self.resize = (320, 320)
#
#     def bmp2npy(self, imgclass):
#         # channel = 3
#         # if (imgclass == "train_label" or imgclass == "test_label"):
#         #     channel = 1
#         channel = 3
#
#         img_npy = np.zeros((self.num, self.height, self.weight, channel))
#         for i in range(self.num):
#             img = cv2.imread(self.path+str(i+1)+'.bmp')
#             img320 = cv2.resize(img, self.resize)
#             if(channel == 1):
#                 img320 = cv2.cvtColor(img320, cv2.COLOR_BGR2GRAY)
#                 img320 = img320[:,:,np.newaxis]
#             img_npy[i] = img320
#             #cv2.imshow("a", img320)
#             #cv2.waitKey(0)
#         # np.save("./1/"+imgclass+".npy", img_npy)
#         np.save('/home/hxq/caffe_test/test-dataset-320/blur100gt/1/gt.npy', img_npy)
#
#     def npycheck(self, name):
#         img = np.load(self.path + name)
#         print(img.shape)
#         for i in range(self.num):
#             cv2.imshow("npycheck", img[i])
#             print(img[i])
#             cv2.waitKey(0)
#
#     def run(self):
#         self.bmp2npy("train")
#         self.bmp2npy("train_label")
#         self.npycheck("train.npy")
#         self.npycheck("train_label.npy")
#
# if __name__ == "__main__":
#     prepocessing().run()
#
#









