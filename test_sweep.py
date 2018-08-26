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
import numpy as np
import progressbar as pb
import shutil

import datetime as dt

if __name__ == '__main__':
    # embed()

    # opt = TestOptions().parse()
    opt = TrainOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.display_id = -1  # no visdom display
    opt.dataroot = './dataset/ilsvrc2012/val/'
    opt.loadSize = 256
    opt.how_many = 1000
    opt.aspect_ratio = 1.0
    opt.sample_Ps = [6,]

    # num_points = np.round(10**np.arange(-.4,3.1,.4))
    num_points = np.round(10**np.arange(-.1,2.8,.1))
    num_points[0] = 0
    num_points = np.unique(num_points.astype('int'))
    N = len(num_points)

    dataset = torchvision.datasets.ImageFolder(opt.dataroot, 
        transform=transforms.Compose([
            transforms.Resize((opt.loadSize,opt.loadSize)),
            transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=opt.batchSize, shuffle=not opt.serial_batches)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    time = dt.datetime.now()
    str_now = '%02d_%02d_%02d%02d'%(time.month,time.day,time.hour,time.minute)
    
    shutil.copyfile('./checkpoints/%s/latest_net_G.pth'%opt.name,'./checkpoints/%s/%s.pth'%(opt.name,str_now))

    psnrs = np.zeros((opt.how_many,N))

    bar = pb.ProgressBar(max_value=opt.how_many)
    for i, data_raw in enumerate(dataset_loader):
        data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        for nn in range(N):
            # embed()
            data = util.get_colorization_data(data_raw, opt, ab_thresh=0., num_points=num_points[nn])

            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            psnrs[i,nn] = util.calculate_psnr_np(util.tensor2im(visuals['real']),util.tensor2im(visuals['fake_reg']))

        # if i % 5 == 0:
            # print('processing (%04d)-th image...'%i)

        if i==opt.how_many-1:
            break

        bar.update(i)


psnrs_mean = np.mean(psnrs, axis=0)
psnrs_std = np.std(psnrs, axis=0)/np.sqrt(opt.how_many)

np.save('psnrs_mean_%s'%str_now,psnrs_mean)
np.save('psnrs_std_%s'%str_now,psnrs_std)
np.save('psnrs_%s'%str_now,psnrs)
print(', ').join(['%.2f'%psnr for psnr in psnrs_mean])

# import matplotlib.pyplot as plt
# plt.plot(num_points,psnrs_mean,'bo-')
# plt.xscale('log')
# plt.savefig('tmp.png')

# print('points: %.2f+/-%.2f'%(mean_points,std_points))
# embed()

# print calculate_psnr_np(util.tensor2im(visuals['real']),util.tensor2im(visuals['fake_reg']))
# print calculate_psnr_torch(visuals['real'],visuals['fake_reg'])

# old_psnrs = [[np.load('psnrs_mean_08_19_2000.npy'),'08_19_2000'],
#     [np.load('psnrs_mean_08_20_0000.npy'),'08_20_0000']]

# embed()

# LOAD_DIR = '/data/big/rzhang/src/pix2pix_stroke/tests_auto/random'
# old_results = np.concatenate([np.load('%s/default_random_0_caffe_%04d_%04d_psnrs.npy'%(LOAD_DIR,a,a+100)) for a in range(0,1000,100)])
old_results = np.load('./psnrs_siggraph.npy')
old_mean = np.mean(old_results, axis=0)
old_std = np.std(old_results, axis=0)/np.sqrt(old_results.shape[0])
print(', ').join(['%.2f'%psnr for psnr in old_mean])

num_points_hack = 1.*num_points
num_points_hack[0] = .4

import matplotlib.pyplot as plt
# plt.close('all')
# for (old_psnr,oo) in old_psnrs:
    # plt.plot(num_points_hack,old_psnr,'k-',label=oo)
    # plt.plot([num_points_hack[0],num_points_hack[-1]],[old_psnr[0],old_psnr[0]],'k-',label='%s (auto)'%oo)

plt.plot(num_points_hack,psnrs_mean,'bo-',label='new')
plt.plot(num_points_hack,psnrs_mean+psnrs_std,'b--')
plt.plot(num_points_hack,psnrs_mean-psnrs_std,'b--')
# plt.plot([num_points_hack[0],num_points_hack[-1]],[psnrs_mean[0],psnrs_mean[0]],'b-',label='new (auto)')
plt.plot(num_points_hack,old_mean,'ro-',label='siggraph17')
plt.plot(num_points_hack,old_mean+old_std,'r--')
plt.plot(num_points_hack,old_mean-old_std,'r--')
# plt.plot([num_points_hack[0],num_points_hack[-1]],[old_mean[0],old_mean[0]],'r-',label='siggraph17 (auto)')

plt.xlabel('Number of points')
plt.ylabel('PSNR [db]')
plt.legend(loc=0)
plt.xscale('log')
plt.xlim((num_points_hack[0],num_points_hack[-1]))
plt.savefig('tmp.png')

# embed()
