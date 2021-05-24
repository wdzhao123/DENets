import torch
from torch import nn
from torch.utils import model_zoo
from torch.nn import functional as F
from san import *

class SFAoatthxq0515(nn.Module):
    def __init__(self):
        super(SFAoatthxq0515, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.dmp = BackEnd()
        self.conv_out = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, input):
        input = self.vgg(input)
        dmp_out_32, dmp_conv5_3_out8 = self.dmp(*input)
        dmp_out_1 = self.conv_out(dmp_out_32)
        return dmp_out_1, dmp_conv5_3_out8

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict, strict=False)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        for p in self.parameters():
            p.requires_grad=False

        for i in range(1,9):
            exec("self.conv5_2_ns_%s = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)" % i)
            exec("self.conv5_3_ns_%s = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)" % i)
            exec("self.conv5_2_bl_%s = BaseConv(512, 512, 1, 1, activation=nn.ReLU(), use_bn=True)" % i)
            exec("self.conv5_3_bl_%s = BaseConv(512, 512, 1, 1, activation=nn.ReLU(), use_bn=True)" % i)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        a = input
        conv5_3_out8 = []
        for i in range(1, 9):
            exec('conv5_2_ns_%s = self.conv5_2_ns_%s(a)'%(i,i))
            exec('conv5_2_blr_%s = (self.conv5_2_bl_%s(a)+conv5_2_ns_%s)/2' % (i, i, i))
            exec('conv5_3_ns_%s = self.conv5_3_ns_%s(conv5_2_ns_%s)' % (i, i, i))
            exec('conv5_3_blr_%s = (self.conv5_3_bl_%s(conv5_2_blr_%s)+conv5_3_ns_%s)/2' % (i, i, i, i))
            eval("conv5_3_out8.append(conv5_3_blr_%s)" % i)
        sum = 0
        num_model = len(conv5_3_out8)
        for j in range(num_model):
            sum += conv5_3_out8[j]
        conv5_3 = sum / num_model
        return conv2_2, conv3_3, conv4_3, conv5_3, conv5_3_out8

    def get_channel(self,input,channel):
        num = len(input)
        f0 = input[0]
        f1 = input[1]
        feature=[]
        for i in range(channel):
            f0i = f0[:, i]
            f1i = f1[:, i]
            f0i = f0i.unsqueeze(1)
            f1i = f1i.unsqueeze(1)
            f_init = torch.cat((f0i, f1i), 1)
            for t in range(2, num):
                fi = input[t]
                fii=fi[:,i]
                fii=fii.unsqueeze(1)
                f_init=torch.cat((f_init,fii),1)
            feature.append(f_init)
        return feature

class NonLocalPartBlockND(nn.Module):
    def __init__(self,
                 in_channels=4096,
                 inter_channels=None,
                 dimension=2,
                 # sub_sample=True,
                 sub_sample=False,
                 bn_layer=True):
        super(NonLocalPartBlockND, self).__init__()
        self.bottle = Bottleneck(0,4096,2048,2048,4096)
        self.conv111 = BaseConv(4096, 512, 1, 1, activation=nn.ReLU(), use_bn=True)



    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv_out8_cat, conv5_3_out8 = input
        x = conv_out8_cat
        z=self.bottle(x)
        z = self.conv111(z)
        return conv2_2, conv3_3, conv4_3, z, conv5_3_out8

class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3, conv5_3_out8 = input#

        input = self.upsample(conv5_3)

        input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.upsample(input)

        input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input, conv5_3_out8


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


if __name__ == '__main__':
    input = torch.randn(4, 3, 400, 400).cuda()
    model = SFAoatthxq0515().cuda()
    output = model(input)
    print(input.size())
    print(output.size())

