import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .vgg import Vgg16
import random


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        #parser.set_defaults(dataset_mode='aligned')
        #parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # use_gan
        self.use_gan = opt.use_GAN
        self.w_vgg = opt.w_vgg
        self.w_tv = opt.w_tv
        self.w_gan = opt.w_gan
        self.use_condition = opt.use_condition
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.use_condition == 1:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,opt.which_model_netD, opt.n_layers_patchD, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,patch=True)
            else:
                self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.which_model_netD == 'multi':
                self.criterionGAN = networks.GANLoss_multi(use_lsgan=not opt.no_lsgan).to(self.device)
            elif opt.use_wgan:
                self.criterionGAN_p = networks.DiscLossWGANGP()
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # load vgg network
            self.vgg = Vgg16().type(torch.cuda.FloatTensor)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_P)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        #self.real_A = input['A' if AtoB else 'B'].to(self.device)
        #self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A = input.to(self.device)
        #self.real_B = input.to(self.device)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        if self.opt.patchD:
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            self.fake_patch = self.fake_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.real_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.real_A[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])

    def backward_D(self):
        #Fake
        if self.use_condition == 1:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB.detach())
        
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        if self.use_condition == 1:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            real_AB = self.real_B
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN_p.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P*2
        self.loss_D_P.backward()

    def backward_G(self):
        self.loss_G = 0
        # First, G(A) should fake the discriminator
        if self.use_gan == 1:
            if self.use_condition == 1:
                fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            else:
                fake_AB = self.fake_B
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
	        self.loss_G_GAN = 0
        ###############################
        loss_G_A = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.netD_P.forward(self.real_patch)
                
                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                      self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0:   
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P.forward(self.real_patch_1[i])
                    
                    loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) +
                                        self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2
                    
            if not self.opt.D_P_times2:
                self.loss_G += (loss_G_A/float(self.opt.patchD_3 + 1))*0.1
            else:
                self.loss_G += (loss_G_A/float(self.opt.patchD_3 + 1)*2)*0.1
            
        else:
            if not self.opt.D_P_times2:
                self.loss_G += loss_G_A
            else:
                self.loss_G += loss_G_A*2
        ##################################
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) 

	
        # vgg loss
        self.real_B_features = self.vgg(self.real_B)
        self.fake_B_features = self.vgg(self.fake_B)
        self.loss_vgg = self.criterionL1(self.fake_B_features[1], self.real_B_features[1])*1 + self.criterionL1(self.fake_B_features[2], self.real_B_features[2])*1 + self.criterionL1(self.fake_B_features[3], self.real_B_features[3])*1 + self.criterionL1(self.fake_B_features[0], self.real_B_features[0])*1
        
        # TV loss
        diff_i = torch.sum(torch.abs(self.fake_B[:, :, :, 1:] - self.fake_B[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(self.fake_B[:, :, 1:, :] - self.fake_B[:, :, :-1, :]))
        self.tv_loss = (diff_i + diff_j) / (256 * 256)

        self.loss_G += self.loss_G_GAN * self.w_gan + self.loss_G_L1 + self.loss_vgg * self.w_vgg + self.tv_loss * self.w_tv

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        if self.use_gan == 1:
       	    self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            self.optimizer_D.step()
            self.optimizer_D_P.step()
        else:
            self.loss_D_fake = 0	    
            self.loss_D_real = 0

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
