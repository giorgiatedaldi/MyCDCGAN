"""Giorgia Tedaldi 339642"""

import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    """Generator code"""

    def __init__(self, args):
        super(Generator, self).__init__()
        
        # Noise 'nz' as input
        self.noise_block = nn.Sequential(
            # input: (batch_size) x (nz) x (1) x (1) --> default: 128 x 100 x 1 x 1
            # output: (batch_size) x (ngf * 2) x (image_size // 32) x (image_size // 32) --> default: 128 x 128 x 4 x 4
            nn.ConvTranspose2d(args.nz, args.ngf * 2, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True)
         )
        
        # Label as input (this is the condition of the model)
        self.label_block = nn.Sequential(
            # input: (batch_size) x (label_dim) x (1) x (1) --> default: 128 x 3 x 1 x 1
            # output: (batch_size) x (ngf * 2) x (image_size // 32) x (image_size // 32) --> default: 128 x 128 x 4 x 4
            nn.ConvTranspose2d(args.label_dim, args.ngf * 2, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True)
         )

        # The first two block will be concatenated on second dimension before passing this block. For this reason
        # the size of the second dimension is (ngf * 4) = (ngf * 2) + (ngf * 2)
        self.main1 = nn.Sequential(
            # input: (batch_size) x (ngf * 4) x (image_size // 32) x (image_size // 32) --> default: 128 x 256 x 4 x 4
            # output: (batch_size) x (ngf * 2) x (image_size // 16) x (image_size // 16) --> default: 128 x 64 x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True)
        )
        
        self.main2 = nn.Sequential(
            # input:  (batch_size) x (ngf * 2) x (image_size // 16) x (image_size // 16) --> default: 128 x 64 x 8 x 8
            # output: (batch_size) x (ngf) x (image_size // 8) x (image_size // 8) --> default: 128 x 64 x 16 x 16 
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True)
        )

        self.main3 = nn.Sequential(
            # input:  (batch_size) x (ngf) x (image_size // 8) x (image_size // 8) --> default: 128 x 64 x 16 x 16
            # output: (batch_size) x (ngf // 2) x (image_size // 4) x (image_size // 4) --> default: 128 x 32 x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf // 2),
            nn.ReLU(True)
        )

        self.main4 = nn.Sequential(
            # input:  (batch_size) x (ngf // 2) x (image_size // 4) x (image_size // 4) --> default: 128 x 32 x 32 x 32
            # output: (batch_size) x (ngf // 4) x (image_size // 2) x (image_size // 2) --> default: 128 x 16 x 64 x 64
            nn.ConvTranspose2d(args.ngf // 2, args.ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf // 4),
            nn.ReLU(True)
        )

        # The final result is a batch with color images (3 channels) with size 128 x 128 
        self.main5 = nn.Sequential(
            # input:  (batch_size) x (ngf // 4) x (image_size // 2) x (image_size // 2) --> default: 128 x 16 x 64 x 64
            # output: (batch_size) x (nc) x (image_size) x (image_size) --> default: 128 x 3 x 128 x 128
            nn.ConvTranspose2d(args.ngf // 4, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # noise is passed in its corresponding layer
        z_out = self.noise_block(noise)

        # labels are passed in their corresponding layer
        l_out = self.label_block(labels)

        # noise and labels result are concatenated and fed to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) 
        x = self.main1(x)
        x = self.main2(x)
        x = self.main3(x)
        x = self.main4(x)
        x = self.main5(x)
        return x


class Discriminator(nn.Module):
    """Discriminator code"""

    def __init__(self, args):
        super(Discriminator, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input:  (batch_size) x (channels) x (image_size) x (image_size) --> default: 128 x 3 x 128 x 128
            # output: (batch_size) x (ndf // 2) x (image_size // 2) x (image_size // 2) --> default: 128 x 32 x 64 x 64
            nn.Conv2d(args.nc, args.ndf//2, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(
            # input:  (batch_size) x (label_dim) x (image_size) x (image_size) --> default: 128 x 3 x 128 x 128
            # output: (batch_size) x (ndf // 2) x (image_size // 2) x (image_size // 2) default: 128 x 32 x 64 x 64
            nn.Conv2d(args.label_dim, args.ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ) 

        # There are 2 different types of main blocks: one for 'adversarial_loss' and one for 'wesserstain_loss'
        # In the first 4 main block for wasserstain_loss it's used 'InstanceNorm2d' insted of 'BatchNorm2d'. Input
        # and output size are always the same for both main block type.     
        self.main1_a = nn.Sequential(
            # input:  (batch_size) x (ndf) x (image_size // 2) x (image_size // 2) --> default: 128 x 64 x 64 x 64
            # output: (batch_size) x (ndf * 2) x (image_size // 4) x (image_size // 4) --> default: 128 x 128 x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main1_w = nn.Sequential(
            # input:  (batch_size) x (ndf) x (image_size // 2) x (image_size // 2) --> default: 128 x 64 x 64 x 64
            # output: (batch_size) x (ndf * 2) x (image_size // 4) x (image_size // 4) --> default: 128 x 128 x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(args.ndf * 2, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main2_a = nn.Sequential(
            # input:  (batch_size) x (ndf * 2) x (image_size // 4) x (image_size // 4) --> default: 128 x 128 x 32 x 32
            # output: (batch_size) x (ndf * 4) x (image_size // 8) x (image_size // 8) --> default: 128 x 256 x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main2_w = nn.Sequential(
            # input:  (batch_size) x (ndf * 2) x (image_size // 4) x (image_size // 4) --> default: 128 x 128 x 32 x 32
            # output: (batch_size) x (ndf * 4) x (image_size // 8) x (image_size // 8) --> default: 128 x 256 x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(args.ndf * 4, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main3_a = nn.Sequential(
            # input:  (batch_size) x (ndf * 4) x (image_size // 8) x (image_size // 8) --> default: 128 x 256 x 16 x 16
            # output: (batch_size) x (ndf * 8) x (image_size // 16) x (image_size // 16) --> default: 128 x 512 x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main3_w = nn.Sequential(
            # input:  (batch_size) x (ndf * 4) x (image_size // 8) x (image_size // 8) --> default: 128 x 256 x 16 x 16
            # output: (batch_size) x (ndf * 8) x (image_size // 16) x (image_size // 16) --> default: 128 x 512 x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(args.ndf * 8, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main4_a = nn.Sequential(
            # input:  (batch_size) x (ndf * 8) x (image_size // 16) x (image_size // 16) --> default: 128 x 512 x 8 x 8
            # output: (batch_size) x (ndf * 16) x (image_size // 32) x (image_size // 32) --> default: 128 x  1024 x 4 x 4
            nn.Conv2d(args.ndf * 8, args.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main4_w = nn.Sequential(
            # input:  (batch_size) x (ndf * 8) x (image_size // 16) x (image_size // 16) --> default: 128 x 512 x 8 x 8
            # output: (batch_size) x (ndf * 16) x (image_size // 32) x (image_size // 32) --> default: 128 x  1024 x 4 x 4
            nn.Conv2d(args.ndf * 8, args.ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(args.ndf * 16, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main5_a = nn.Sequential(
            # input:  (batch_size) x (ndf * 16) x (image_size // 32) x (image_size // 32) --> default: 128 x  1024 x 4 x 4
            # output: (batch_size) x (1) x (1) x (1)
            nn.Conv2d(args.ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # With wesserstein loss no Sigmoid is used
        self.main5_w = nn.Sequential(
            # input:  (batch_size) x (ndf * 16) x (image_size // 32) x (image_size // 32) --> default: 128 x  1024 x 4 x 4
            # output: (batch_size) x (1) x (1) x (1) --> default: 128 x 1 x 1 x 1
            nn.Conv2d(args.ndf * 16, 1, 4, 1, 0, bias=False),
        )
        
        # The output of the sixth main block is used when classification loss is enabled
        self.main6_a = nn.Sequential(
            # input:  (batch_size) x (ndf * 16) x (image_size // 32) x (image_size // 32) --> default: 128 x  1024 x 4 x 4
            # output: (batch_size) x (label_dim) x (1) x (1) --> default: 128 x 3 x 1 x 1 
            nn.Conv2d(args.ndf * 16, args.label_dim, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.Sigmoid()
        )

        self.main6_w = nn.Sequential(
            # input:  (batch_size) x (ndf * 16) x (image_size // 32) x (image_size // 32) --> default: 128 x  1024 x 4 x 4
            # output: (batch_size) x (label_dim) x (1) x (1) --> default: 128 x 3 x 1 x 1 
            nn.Conv2d(args.ndf * 16, args.label_dim, kernel_size = 4, stride = 1, padding = 0, bias=False),
        )

    def forward(self, img, label, loss):
        # images are passed to their corresponding layer
        img_out = self.img_block(img)

        # labels are passed to their corresponding layer
        lab_out = self.label_block(label)
       
        # images and labels are concatenated and fed to the rest of the discriminator
        x = torch.cat([img_out, lab_out], dim = 1)

        if(loss == 'adversarial_loss'):
            x = self.main1_a(x)
            x = self.main2_a(x)
            x = self.main3_a(x)
            x = self.main4_a(x)
            out_src = self.main5_a(x)
            out_cls = self.main6_a(x)
        elif (loss == 'wasserstein_loss'):
            x = self.main1_w(x)
            x = self.main2_w(x)
            x = self.main3_w(x)
            x = self.main4_w(x)
            out_src = self.main5_w(x)
            out_cls = self.main6_w(x) 

        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src, out_cls
