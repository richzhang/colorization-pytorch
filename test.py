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

if __name__ == '__main__':
    # opt = TestOptions().parse()
    opt = TrainOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.dataroot = '/data/big/dataset/ILSVRC2012/val2/'
    opt.loadSize = 256
    opt.how_many = 200
    opt.aspect_ratio = 1.0

    dataset = torchvision.datasets.ImageFolder(opt.dataroot, 
        transform=transforms.Compose([
            transforms.Resize(opt.loadSize),
            transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=opt.batchSize, shuffle=not opt.serial_batches)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    psnrs_auto = np.zeros(opt.how_many)
    psnrs_points = np.zeros(opt.how_many)

    for i, data_raw in enumerate(dataset_loader):
        data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        # with no points
        img_path = ['%08d_gray'%i,]
        data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=1.)

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        psnrs_auto[i] = util.calculate_psnr_np(util.tensor2im(visuals['real']),util.tensor2im(visuals['fake_reg']))

        # with random points
        img_path = ['%08d_pts'%i,]
        data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=.125)

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        psnrs_points[i] = util.calculate_psnr_np(util.tensor2im(visuals['real']),util.tensor2im(visuals['fake_reg']))

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        if i==opt.how_many-1:
            break

    webpage.save()

mean_auto = np.mean(psnrs_auto)
mean_points = np.mean(psnrs_points)
std_auto = np.std(psnrs_auto)/np.sqrt(opt.how_many)
std_points = np.std(psnrs_points)/np.sqrt(opt.how_many)

print('automatic: %.2f+/-%.2f'%(mean_auto,std_auto))
print('+points: %.2f+/-%.2f'%(mean_points,std_points))
