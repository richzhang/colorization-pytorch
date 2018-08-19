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

from util import util
from IPython import embed

if __name__ == '__main__':
    # opt = TestOptions().parse()
    opt = TrainOptions().parse()

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.dataroot = '/data/big/dataset/ILSVRC2012/val2/'

    opt.loadSize = 256
    # opt.fineSize = 256

    dataset = torchvision.datasets.ImageFolder(opt.dataroot, 
        transform=transforms.Compose([
            transforms.Resize(opt.loadSize),
            transforms.ToTensor()]))
            # transforms.RandomResizedCrop(opt.fineSize),
            # transforms.RandomHorizontalFlip(),
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=opt.batchSize, shuffle=not opt.serial_batches)

    # embed()

    # opt.how_many = 200
    opt.how_many = 168
    opt.aspect_ratio = 1.0

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    for i, data_raw in enumerate(dataset_loader):
        img_path = ['%08d_gray'%i,]
        data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=128)
        data = util.get_colorization_data(data_raw, l_norm=opt.l_norm, ab_norm=opt.ab_norm, l_cent=opt.l_cent, mask_cent=opt.mask_cent, ab_thresh=0., p=1.)

        # break
        # embed()
        # data_ab = data['B']
        # data_ind = util.encode_ab_ind(data_ab)
        # data_ab_recon = util.decode_ind_ab(data_ind)
        # embed()

        # with no points
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        # with random points
        img_path = ['%08d_pts'%i,]
        data = util.get_colorization_data(data_raw, l_norm=opt.l_norm, ab_norm=opt.ab_norm, l_cent=opt.l_cent, mask_cent=opt.mask_cent, ab_thresh=0., p=.125)

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        if i==opt.how_many-1:
            break

    webpage.save()

embed()

def add_color(data,mask,mask_cent=.5,hw=[128,128],P=1,ab=[0,0]):
    data[:,0,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[0]/opt.ab_norm
    data[:,1,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[1]/opt.ab_norm
    mask[:,:,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.-mask_cent

    return (data,mask)

import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

img = plt.imread('%s/val/ILSVRC2012_val_00000168.JPEG'%opt.dataroot)/255.
zoom_factor = 256./img.shape[0]
img = zoom(img,[zoom_factor, zoom_factor, 1])
data_raw = [util.crop_mult(torch.Tensor(img.transpose((2,0,1)))[None,:,:,:]),]
data_raw[0] = util.crop_mult(data_raw[0],mult=128).cuda()

# data_raw[0] = data_raw[0]
data_raw[0] = util.crop_mult(data_raw[0], mult=128)
data = util.get_colorization_data(data_raw, l_norm=opt.l_norm, ab_norm=opt.ab_norm, l_cent=opt.l_cent, mask_cent=opt.mask_cent, ab_thresh=0., p=1.)
data['hint_B'],data['mask_B'] = add_color(data['hint_B'],data['mask_B'],ab=[40,0],P=5,hw=[128,164])

model.set_input(data)
model.test()
visuals = model.get_current_visuals()

for key in visuals.keys():
    plt.imsave('tmp_%s.png'%key, util.tensor2im(visuals[key]))


import numpy as np
weights_l = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0))[:,:,0,:].reshape((3,3,8,8))
weights_a = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0))[:,:,1,:].reshape((3,3,8,8))
weights_b = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0))[:,:,2,:].reshape((3,3,8,8))
weights_m = model.netG.module.model1[0].weight.cpu().data.numpy().transpose((2,3,1,0))[:,:,3,:].reshape((3,3,8,8))

print np.mean(np.abs(weights_l))
print np.mean(np.abs(weights_a))
print np.mean(np.abs(weights_b))
print np.mean(np.abs(weights_m))

# mont = np.zeros((32,32,3))
# for a in range(8):
#     for b in range(8):
#         mont[4*a:4*a+3, 4*b:4*b+3, :] = (weights[:,:,a,b]-np.mean(weights[:,:,a,b]))/np.std(weights[:,:,a,b])
# plt.imsave('weights,png',mont)

