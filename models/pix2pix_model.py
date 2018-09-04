import torch
from collections import OrderedDict
from util.image_pool import ImagePool
from util import util
from .base_model import BaseModel
from . import networks
import numpy as np


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.half = opt.half

        self.use_D = self.opt.lambda_GAN > 0

        # specify the training losses you want to print out. The program will call base_model.get_current_losses

        if(self.use_D):
            self.loss_names = ['G_GAN', ]
        else:
            self.loss_names = []

        self.loss_names += ['G_CE', 'G_entr', 'G_entr_hint', ]
        self.loss_names += ['G_L1_max', 'G_L1_mean', 'G_entr', 'G_L1_reg', ]
        self.loss_names += ['G_fake_real', 'G_fake_hint', 'G_real_hint', ]
        self.loss_names += ['0', ]

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        if self.isTrain:
            if(self.use_D):
                self.model_names = ['G', 'D']
            else:
                self.model_names = ['G', ]
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        num_in = opt.input_nc + opt.output_nc + 1
        self.netG = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      use_tanh=True, classification=opt.classification)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.use_D:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1 = networks.L1Loss()
            self.criterionHuber = networks.HuberLoss(delta=1. / opt.ab_norm)

            # if(opt.classification):
            self.criterionCE = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if self.use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

        if self.half:
            for model_name in self.model_names:
                net = getattr(self, 'net' + model_name)
                net.half()
                for layer in net.modules():
                    if(isinstance(layer, torch.nn.BatchNorm2d)):
                        layer.float()
                print('Net %s half precision' % model_name)

        # initialize average loss values
        self.avg_losses = OrderedDict()
        self.avg_loss_alpha = opt.avg_loss_alpha
        self.error_cnt = 0

        # self.avg_loss_alpha = 0.9993 # half-life of 1000 iterations
        # self.avg_loss_alpha = 0.9965 # half-life of 200 iterations
        # self.avg_loss_alpha = 0.986 # half-life of 50 iterations
        # self.avg_loss_alpha = 0. # no averaging
        for loss_name in self.loss_names:
            self.avg_losses[loss_name] = 0

    def set_input(self, input):
        if(self.half):
            for key in input.keys():
                input[key] = input[key].half()

        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.hint_B = input['hint_B'].to(self.device)
        self.mask_B = input['mask_B'].to(self.device)
        self.mask_B_nc = self.mask_B + self.opt.mask_cent

        self.real_B_enc = util.encode_ab_ind(self.real_B[:, :, ::4, ::4], self.opt)

    def forward(self):
        (self.fake_B_class, self.fake_B_reg) = self.netG(self.real_A, self.hint_B, self.mask_B)
        # if(self.opt.classification):
        self.fake_B_dec_max = self.netG.module.upsample4(util.decode_max_ab(self.fake_B_class, self.opt))
        self.fake_B_distr = self.netG.module.softmax(self.fake_B_class)

        self.fake_B_dec_mean = self.netG.module.upsample4(util.decode_mean(self.fake_B_distr, self.opt))

        self.fake_B_entr = self.netG.module.upsample4(-torch.sum(self.fake_B_distr * torch.log(self.fake_B_distr + 1.e-10), dim=1, keepdim=True))
        # embed()

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # self.loss_D_fake = 0

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # self.loss_D_real = 0

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def compute_losses_G(self):
        mask_avg = torch.mean(self.mask_B_nc.type(torch.cuda.FloatTensor)) + .000001

        self.loss_0 = 0  # 0 for plot

        # classification statistics
        self.loss_G_CE = self.criterionCE(self.fake_B_class.type(torch.cuda.FloatTensor),
                                          self.real_B_enc[:, 0, :, :].type(torch.cuda.LongTensor))  # cross-entropy loss
        self.loss_G_entr = torch.mean(self.fake_B_entr.type(torch.cuda.FloatTensor))  # entropy of predicted distribution
        self.loss_G_entr_hint = torch.mean(self.fake_B_entr.type(torch.cuda.FloatTensor) * self.mask_B_nc.type(torch.cuda.FloatTensor)) / mask_avg  # entropy of predicted distribution at hint points

        # regression statistics
        self.loss_G_L1_max = 10 * torch.mean(self.criterionL1(self.fake_B_dec_max.type(torch.cuda.FloatTensor),
                                                              self.real_B.type(torch.cuda.FloatTensor)))
        self.loss_G_L1_mean = 10 * torch.mean(self.criterionL1(self.fake_B_dec_mean.type(torch.cuda.FloatTensor),
                                                               self.real_B.type(torch.cuda.FloatTensor)))
        self.loss_G_L1_reg = 10 * torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                              self.real_B.type(torch.cuda.FloatTensor)))

        # L1 loss at given points
        self.loss_G_fake_real = 10 * torch.mean(self.criterionL1(self.fake_B_reg * self.mask_B_nc, self.real_B * self.mask_B_nc).type(torch.cuda.FloatTensor)) / mask_avg
        self.loss_G_fake_hint = 10 * torch.mean(self.criterionL1(self.fake_B_reg * self.mask_B_nc, self.hint_B * self.mask_B_nc).type(torch.cuda.FloatTensor)) / mask_avg
        self.loss_G_real_hint = 10 * torch.mean(self.criterionL1(self.real_B * self.mask_B_nc, self.hint_B * self.mask_B_nc).type(torch.cuda.FloatTensor)) / mask_avg

        # self.loss_G_L1 = torch.mean(self.criterionL1(self.fake_B, self.real_B))
        # self.loss_G_Huber = torch.mean(self.criterionHuber(self.fake_B, self.real_B))
        # self.loss_G_fake_real = torch.mean(self.criterionHuber(self.fake_B*self.mask_B_nc, self.real_B*self.mask_B_nc)) / mask_avg
        # self.loss_G_fake_hint = torch.mean(self.criterionHuber(self.fake_B*self.mask_B_nc, self.hint_B*self.mask_B_nc)) / mask_avg
        # self.loss_G_real_hint = torch.mean(self.criterionHuber(self.real_B*self.mask_B_nc, self.hint_B*self.mask_B_nc)) / mask_avg

        if self.use_D:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G = self.loss_G_CE * self.opt.lambda_A + self.loss_G_L1_reg
            # self.loss_G = self.loss_G_Huber*self.opt.lambda_A

    def backward_G(self):
        self.compute_losses_G()
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        if(self.use_D):
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

            self.set_requires_grad(self.netD, False)

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()

        visual_ret['gray'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), torch.zeros_like(self.real_B).type(torch.cuda.FloatTensor)), dim=1), self.opt)
        visual_ret['real'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)

        visual_ret['fake_max'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.fake_B_dec_max.type(torch.cuda.FloatTensor)), dim=1), self.opt)
        visual_ret['fake_mean'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.fake_B_dec_mean.type(torch.cuda.FloatTensor)), dim=1), self.opt)
        visual_ret['fake_reg'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)

        visual_ret['hint'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.hint_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)

        visual_ret['real_ab'] = util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)

        visual_ret['fake_ab_max'] = util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.fake_B_dec_max.type(torch.cuda.FloatTensor)), dim=1), self.opt)
        visual_ret['fake_ab_mean'] = util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.fake_B_dec_mean.type(torch.cuda.FloatTensor)), dim=1), self.opt)
        visual_ret['fake_ab_reg'] = util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)

        visual_ret['mask'] = self.mask_B_nc.expand(-1, 3, -1, -1).type(torch.cuda.FloatTensor)
        visual_ret['hint_ab'] = visual_ret['mask'] * util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.hint_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)

        C = self.fake_B_distr.shape[1]
        # scale to [-1, 2], then clamped to [-1, 1]
        visual_ret['fake_entr'] = torch.clamp(3 * self.fake_B_entr.expand(-1, 3, -1, -1) / np.log(C) - 1, -1, 1)

        return visual_ret

    # return training losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        self.error_cnt += 1
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                self.avg_losses[name] = float(getattr(self, 'loss_' + name)) + self.avg_loss_alpha * self.avg_losses[name]
                errors_ret[name] = (1 - self.avg_loss_alpha) / (1 - self.avg_loss_alpha**self.error_cnt) * self.avg_losses[name]

        # errors_ret['|ab|_gt'] = float(torch.mean(torch.abs(self.real_B[:,1:,:,:])).cpu())
        # errors_ret['|ab|_pr'] = float(torch.mean(torch.abs(self.fake_B[:,1:,:,:])).cpu())

        return errors_ret
