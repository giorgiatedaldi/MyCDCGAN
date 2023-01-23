import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class ResidualBlock(nn.Module):
#     """Residual Block with instance normalization."""
#     def __init__(self, dim_in, dim_out):
#         super(ResidualBlock, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

#     def forward(self, x):
#         return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input: (batch_size) x (nz) x (1) x (1)
            # output: (batch_size) x (ngf * 2) x (4) x (4)
            nn.ConvTranspose2d(args.nz, args.ngf * 2, 4, 1, 0, bias=False), # bs, 100, 1, 1 -> bs, ngf * 2, 4, 4
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input: (batch_size) x (label_dim) x (1) x (1)
            # output: (batch_size) x (ngf * 2) x (4) x (4)
            nn.ConvTranspose2d(args.label_dim, args.ngf * 2, 4, 1, 0, bias=False), #bs, 10, 1, 1 -> bs, ngf * 2, 4, 4
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True)
         )

        self.main1 = nn.Sequential(
            # input: (batch_size) x (ngf * 4) x (4) x (4)
            # output: (batch_size) x (ngf * 2) x (8) x (8)
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True)
        )
        
        self.main2 = nn.Sequential(
            # input: (batch_size) x (ngf * 2) x (8) x (8)
            # output: (batch_size) x (ngf) x (16) x (16)
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True)
        )

        self.main3 = nn.Sequential(
            # input: (batch_size) x (ngf * 2) x (8) x (8)
            # output: (batch_size) x (ngf) x (16) x (16)
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(args.ngf, args.ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf // 2),
            nn.ReLU(True)
        )

        self.main4 = nn.Sequential(
            # input: (batch_size) x (ngf * 2) x (8) x (8)
            # output: (batch_size) x (ngf) x (16) x (16)
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(args.ngf // 2, args.ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf // 4),
            nn.ReLU(True)
        )


        self.main5 = nn.Sequential(
            # input: (batch_size) x (ngf) x (8) x (8)
            # output: (batch_size) x (ngf * 2) x (4) x (4)
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(args.ngf // 4, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        #print('-----------z_out-----------------------')
        #print(z_out.size())
        
        l_out = self.label_block(labels)
        # print('------------l_out----------------------')
        # print(l_out.size())
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        # bs, ngf*4, 4, 4
        # print('-------------catimglabel---------------------')
        # print(x.size())

        x = self.main1(x)
        # print('--------------main1G-----------------')
        # print(x.size())

        x = self.main2(x)
        # print('-------------------main2g---------------')
        # print(x.size())

        x = self.main3(x)
        # print('------------------main3g----------------')
        # print(x.size())

        x = self.main4(x)
        # print('------------------main4g----------------')
        # print(x.size())

        x = self.main5(x)
        # print('------------------main5g----------------')
        # print(x.size())



        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input: (batch_size) x (channels) x (image_size) x (image_size)
            # output: (batch_size) x (ndf/2) x (image_size / 2) x (image_size / 2)
            nn.Conv2d(args.nc, args.ndf//2, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(
            # input: (batch_size) x (label_dim) x (image_size) x (image_size)
            # output: (batch_size) x (ndf/2) x (image_size / 2) x (image_size / 2)
            nn.Conv2d(args.label_dim, args.ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main1 = nn.Sequential(
            # input: (batch_size) x (ndf) x (image_size / 2) x (image_size / 2)
            # output: (batch_size) x (ndf * 2) x (image_size / 4) x (image_size / 4)
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main2 = nn.Sequential(
            # input: (batch_size) x (ndf * 2) x (image_size / 4) x (image_size / 4)
            # output: (batch_size) x (ndf * 4) x (image_size / 8) x (image_size / 8)
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main3 = nn.Sequential(
            # input: (batch_size) x (ndf * 4) x (image_size / 8) x (image_size / 8)
            # output: (batch_size) x (ndf * 8) x (image_size / 16) x (image_size / 16)
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main4 = nn.Sequential(
            # input: (batch_size) x (ndf * 8) x (image_size / 16) x (image_size / 16)
            # output: (batch_size) x (ndf * 16) x (image_size / 32) x (image_size / 32)
            nn.Conv2d(args.ndf * 8, args.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main5 = nn.Sequential(
            # input: (batch_size) x (ndf * 16) x (image_size / 32) x (image_size / 32)
            # output: (batch_size) x (1) x (1) x (1) ------> (16x1x1x1)
            nn.Conv2d(args.ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1 x 1 x 1
        )

        

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        # print('-----------img_out-----------------------')
        # print(img_out.size())

        lab_out = self.label_block(label)
        # print('----------------lab_out------------------')
        # print(lab_out.size())

        x = torch.cat([img_out, lab_out], dim = 1)
        # print('----------------catD------------------')
        # print(x.size())

        x = self.main1(x)
        # print('-------------main1D---------------------')
        # print(x.size())


        x = self.main2(x)
        # print('------------main2d----------------------')
        # print(x.size())


        x = self.main3(x)
        # print('----------------main3d------------------')
        # print(x.size())

        x = self.main4(x)
        # print('----------------main4d------------------')
        # print(x.size())

        x = self.main5(x)
        # print('----------------main4d------------------')
        # print(x.size())

        return x
