
from util import util
from IPython import embed

import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

# import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import cv2
import scipy

embed()

opt = TrainOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.display_id = -1  # no visdom display
opt.dataroot = './dataset/ilsvrc2012/val/'
opt.loadSize = 256
opt.how_many = 1000
opt.aspect_ratio = 1.0
opt.sample_Ps = [6, ]

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.display_id = -1  # no visdom display
# opt.dataroot = './datasets/ilsvrc2012/val/val/ILSVRC2012_val_00000168.JPEG'
opt.loadSize = 256
opt.load_model = True

# # data loader
# for i, data_raw_ in enumerate(dataset_loader):
#     data_raw_[0] = data_raw_[0].cuda()
#     data_raw_[0] = util.crop_mult(data_raw_[0], mult=8)

#     # with no points
#     img_path = ['%08d_gray'%i,]
#     data_ = util.get_colorization_data(data_raw_, opt, ab_thresh=0., p=1.)

#     if i % 5 == 0:
#         print('processing (%04d)-th image... %s' % (i, img_path))

#     if i==opt.how_many-1:
#         break

# model.set_input(data_)
# model.test()
# visuals = model.get_current_visuals()
# for key in visuals.keys():
#     plt.imsave('cup2_%s.png'%key, util.tensor2im(visuals[key]))


model = create_model(opt)
model.setup(opt)
model.eval()

img = cv2.imread('%s/val/ILSVRC2012_val_00000168.JPEG' % opt.dataroot)[:, :, ::-1]
# zoom_factor0 = 256./img.shape[0]
# zoom_factor1 = 256./img.shape[1]
# img = zoom(img,[zoom_factor0, zoom_factor1, 1], order=1)
img = scipy.misc.imresize(img, (256, 256)) / 255.
# img = img[::2,::2,:]
data_raw = [util.crop_mult(torch.Tensor(img.transpose((2, 0, 1)))[None, :, :, :]), ]
data_raw[0] = util.crop_mult(data_raw[0], mult=8).cuda()

data_raw[0] = data_raw[0]
# data_raw[0] = util.crop_mult(data_raw[0], mult=8)
data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=1.)
data['hint_B'], data['mask_B'] = util.add_color_patch(data['hint_B'], data['mask_B'], opt, ab=[60, 0], P=5, hw=[136, 164])
data['hint_B'], data['mask_B'] = util.add_color_patch(data['hint_B'], data['mask_B'], opt, ab=[60, 0], P=5, hw=[170, 164])
data['hint_B'], data['mask_B'] = util.add_color_patch(data['hint_B'], data['mask_B'], opt, ab=[60, 0], P=5, hw=[196, 164])

model.set_input(data)
model.test()
visuals = model.get_current_visuals()

for key in visuals.keys():
    cv2.imwrite('cup_%s.png' % key, util.tensor2im(visuals[key])[:, :, ::-1])


# import numpy as np
# weights_l = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0))[:,:,0,:].reshape((3,3,8,8))
# weights_a = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0))[:,:,1,:].reshape((3,3,8,8))
# weights_b = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0))[:,:,2,:].reshape((3,3,8,8))
# weights_m = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0))[:,:,3,:].reshape((3,3,8,8))
# weights = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0)).reshape((3,3,4,64))
# print np.mean(np.mean(np.mean(np.abs(weights),axis=3),axis=0),axis=0)


# mont = np.zeros((32,32,3))
# for a in range(8):
#     for b in range(8):
#         mont[4*a:4*a+3, 4*b:4*b+3, :] = (weights_l[:,:,a,b]-np.mean(weights_l[:,:,a,b]))/np.std(weights_l[:,:,a,b])
# plt.imsave('weights.png',mont)
