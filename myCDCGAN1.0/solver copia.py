from model import Generator
from model import Discriminator
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
import torch
import torch.nn.functional as F
import numpy as np
import os
import time

import datetime

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Solver(object):
    """Solver for training and testing StarGAN."""
    def __init__(self, dataloader, args, writer):
        """Initialize configurations."""
        self.dataloader = dataloader
        self.args = args
        self.writer = writer

        # # Data loader.
        # self.dataloader = dataloader
        # # Model configurations.
        # self.label_dim = args.c_dim
        # self.image_size = args.image_size
        # self.g_conv_dim = args.g_conv_dim
        # self.d_conv_dim = args.d_conv_dim
        # self.g_repeat_num = args.g_repeat_num
        # self.d_repeat_num = args.d_repeat_num
        # self.nz = args.nz
        # self.ngf = args.ngf
        # self.ndf = args.ndf
        # self.nc = args.nc
        # #self.lambda_cls = args.lambda_cls
        # #self.lambda_rec = args.lambda_rec
        # #self.lambda_gp = args.lambda_gp

        # # Training configurations.
        # self.batch_size = args.batch_size
        # # self.num_iters = args.num_iters
        # # self.num_iters_decay = args.num_iters_decay
        # # self.g_lr = args.g_lr
        # # self.d_lr = args.d_lr
        # # self.n_critic = args.n_critic
        # # self.beta1 = args.beta1
        # # self.beta2 = args.beta2
        # # self.resume_iters = args.resume_iters
        # self.selected_attrs = args.selected_attrs

        # # Test configurations.
        # #self.test_iters = args.test_iters

        # # Miscellaneous.
        # self.use_tensorboard = args.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # #MODE E NUM WORKERS?????

        # # Directories.
        # self.log_dir = args.log_dir
        # self.sample_dir = args.sample_dir
        # self.model_save_dir = args.model_save_dir
        # self.result_dir = args.result_dir

        # # Step size.
        # self.log_step = args.log_step
        # self.sample_step = args.sample_step
        # self.model_save_step = args.model_save_step
        # self.lr_update_step = args.lr_update_step

        # Build the model and tensorboard.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.args).to(self.device)
        self.D = Discriminator(self.args).to(self.device) 
        
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.apply(weights_init)
        self.D.apply(weights_init)

        if(self.args.loss == 'adversarial_loss'):
            self.criterion = nn.BCELoss()
        elif(self.args.loss == 'wesserstein_loss'):
            self.criterion = 0

        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.args.g_lr, [self.args.beta1, self.args.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.args.d_lr, [self.args.beta1, self.args.beta2])


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def load_model(self, resume_epochs, iter):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from iter [{}/{}] of epoch [{}/{}]...'.format(iter, len(self.dataloader), resume_epochs, self.args.epochs))
        G_path = os.path.join(self.args.model_save_dir, self.args.model_name,'iter_{}_epoch_{}-G.ckpt'.format(iter, resume_epochs))
        D_path = os.path.join(self.args.model_save_dir, self.args.model_name,'iter_{}_epoch_{}-D.ckpt'.format(iter, resume_epochs))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def save_model (self,resume_epochs, iter):
        """Store the trained generator and discriminator."""
        G_path = os.path.join(self.args.model_save_dir, self.args.model_name, 'iter_{}_epoch_{}-G.ckpt'.format(iter, resume_epochs))
        D_path = os.path.join(self.args.model_save_dir, self.args.model_name, 'iter_{}_epoch_{}-D.ckpt'.format(iter, resume_epochs))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(self.args.model_save_dir))
    
    def train(self):
        img_list = []
        G_losses = []
        D_losses = []

        # Start training from scratch or resume training.
        start_epoch = 1
        start_iter = 1
        if self.args.resume_epoch and self.args.resume_iter:
            start_epoch = self.args.resume_epoch
            start_iter = self.args.resume_iter
            self.load_model(self.args.resume_epoch, self.args.resume_iter)

        # Start training.
        print('Start training...')
        start_time = time.time()
        onehot = torch.ones(self.args.label_dim, self.args.label_dim)
        #print(len(self.dataloader))
        #print(self.args.num_samples)
        for epoch in range (start_epoch, self.args.epochs + 1):

            print('-----------STARTING EPOCH [{}/{}]-----------'.format(epoch, self.args.epochs))

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            for i, (images, labels) in enumerate(self.dataloader, 0):
                
                if(i == len(self.dataloader) - start_iter):
                    break

                # self.save_model(epoch, i + start_iter)
                # log = " Iteration [{}/{}] on epoch [{}/{}]".format(start_iter+i, len(self.dataloader), epoch, self.args.epochs)
                # print(log)
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.D.zero_grad()
                b_size = images.size(0)

                # print('---------IMG SIZE --------')
                # print(images.size())

                # print('---------label SIZE --------')
                # print(labels.size())

               

                broadcasted_labels = torch.zeros(b_size, self.args.label_dim, self.args.image_size, self.args.image_size, device = self.device)
                g_labels = labels.unsqueeze(-1).unsqueeze(-1).to(self.device)
                d_labels = broadcasted_labels + g_labels

                # print('---------------GALABELS-------------')
                # print(g_labels.size())
                # print('---------------DALABELS-------------')
                # print(d_labels.size())
                
                # Establish convention for real and fake labels during training
                # Let's do it more simply than last time
                #real_label = torch.ones(b_size).to(self.device)
                #fake_label = torch.zeros(b_size).to(self.device)

                real_label = torch.full((b_size,), 1, dtype=torch.float, device=self.device)
                fake_label = torch.full((b_size,), 0, dtype=torch.float, device=self.device)
                
                # Format batch
                real_cpu = images.to(self.device)

                # print('---------IMG SIZE after to device--------')
                # print(real_cpu.size())

                #c_fill = fill[labels].to(self.device)
                # Forward pass real batch through D
                d_output_real = self.D(real_cpu, d_labels).view(-1) #????????????????????????????'
                # Calculate loss on all-real batch
                # print('----------D_OUTPUT_REAL-------------')
                # print(d_output_real.size())
                # print('----------REAL_LABEL-------------')
                # print(real_label.size())
                errD_real = self.criterion(d_output_real, real_label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = d_output_real.mean().item()
                accuracy_real = torch.mean(1 - torch.abs(d_output_real - real_label))

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.args.nz, 1, 1, device = self.device)
                # pick random labels ang generate corresponding onehot
                #c = (torch.rand(b_size) * self.label_dim).type(torch.LongTensor) #equivalent to int64 #[0,6,4,3,9]
                #c_onehot = onehot[c].to(self.device)
                # Generate fake image batch with G
                fake = self.G(noise, g_labels)
                # Classify all fake batch with D
                #c_fill = fill[c].to(device)
                d_output_fake = self.D(fake.detach(), d_labels).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(d_output_fake, fake_label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = d_output_fake.mean().item()
                accuracy_fake = torch.mean(1 - torch.abs(d_output_fake - fake_label))
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.d_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.G.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                g_output = self.D(fake, d_labels).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(g_output, real_label) # fake images are real for generator cost
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = g_output.mean().item()
                # Update G
                self.g_optimizer.step()

                loss = {}
                loss['D/loss_real'] = errD_real.item()
                loss['D/loss_fake'] = errD_fake.item()
                loss['D/loss'] = errD.item()
                loss['G/loss'] = errG.item()

                accuracy = {}
                accuracy['D/accuracy_real'] = accuracy_real.item()
                accuracy['D/accuracy_fake'] = accuracy_fake.item()
                


                fixed_noise = torch.randn(8, self.args.nz, 1, 1, device=self.device)

                ###################################################################

                # Output training stats
                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                
                # Print out training information.
                if (i+1) % self.args.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}] on epoch [{}/{}]".format(et, i+start_iter, len(self.dataloader), epoch, self.args.epochs)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    for tag, value in loss.items():
                        #self.logger.scalar_summary(tag, value, i+1)
                        self.writer.add_scalar(tag, value, i+start_iter)
                    
                    for tag, value in accuracy.items():
                        #self.logger.scalar_summary(tag, value, i+1)
                        self.writer.add_scalar(tag, value, i+start_iter)
                if ((i+1) % self.args.sample_step == 0) or (epoch == self.args.epochs):
                    # Check how the generator is doing by saving G's output on fixed_noise
                    with torch.no_grad():
                        condition = torch.ones(g_labels.size()).to(self.device)
                        fake = self.generate_test(fixed_noise, condition, self.G).detach().cpu()
                    im_grid = vutils.make_grid(fake, padding=2, normalize=True)
                    img_list.append(im_grid)
                    #sample_path = os.path.join(self.args.sample_dir, self.args.model_name, '{}-images.jpg'.format(i+1))
                    #vutils.save_image(im_grid, sample_path)
                    self.writer.add_image('samples_{}/_{}/_iter_{}_epoch_{}'.format(self.args.model_name, self.args.run_name, i+start_iter, epoch), im_grid, start_iter + 1)
                    #print(i)
                
                if(i+1) % self.args.model_save_step == 0:
                    self.save_model(epoch, i + start_iter)


            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            print('TRAINING TIME FOR EPOCH [{}/{}]: {}'.format(epoch, self.args.epochs, et))
            self.save_model(epoch, i + start_iter)
        

        self.writer.flush()
        self.writer.close()

    # function to generate test sample
    def generate_test(self, fixed_noise, condition, G):
        G.eval()
        # label 0
        c = (torch.ones(8)*0).type(torch.LongTensor) #[0,0,0,0,0,0,0,0]
        c_onehot = condition[c].to(self.device)
        out = G(fixed_noise, c_onehot)
        #inference_res = out

        # labels for all selected attributes    
        #for l in range(1,self.args.label_dim):
            #c = (torch.ones(8)*l).type(torch.LongTensor)
            #c_onehot = onehot[c].to(self.device)
            #out = G(fixed_noise, c_onehot)
            #inference_res = torch.cat([inference_res, out], dim = 0)
        G.train()
        return out
