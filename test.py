
import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html

import string
import torch
import torchvision
import torchvision.transforms as transforms

from util import util
import numpy as np


if __name__ == '__main__':
    sample_ps = [1., .125, .03125]
    to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
    S = len(sample_ps)

    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'
    opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
    opt.serial_batches = True
    opt.aspect_ratio = 1.

    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # statistics
    psnrs = np.zeros((opt.how_many, S))
    entrs = np.zeros((opt.how_many, S))

    for i, data_raw in enumerate(dataset_loader):
        data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        # with no points
        for (pp, sample_p) in enumerate(sample_ps):
            img_path = [string.replace('%08d_%.3f' % (i, sample_p), '.', 'p')]
            data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=sample_p)

            model.set_input(data)
            model.test(True)  # True means that losses will be computed
            visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

            psnrs[i, pp] = util.calculate_psnr_np(util.tensor2im(visuals['real']), util.tensor2im(visuals['fake_reg']))
            entrs[i, pp] = model.get_current_losses()['G_entr']

            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        if i == opt.how_many - 1:
            break

    webpage.save()

    # Compute and print some summary statistics
    psnrs_mean = np.mean(psnrs, axis=0)
    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

    entrs_mean = np.mean(entrs, axis=0)
    entrs_std = np.std(entrs, axis=0) / np.sqrt(opt.how_many)

    for (pp, sample_p) in enumerate(sample_ps):
        print('p=%.3f: %.2f+/-%.2f' % (sample_p, psnrs_mean[pp], psnrs_std[pp]))
