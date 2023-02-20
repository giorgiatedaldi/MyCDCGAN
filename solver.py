"""Giorgia Tedaldi 339642"""

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
import random
import datetime

def weights_init(m):
    """custom weights initialization called on netG and netD: weights are randomly 
    initialized from a Normal distribution with mean=0, stdev=0.02"""
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Solver(object):
    """Solver for training and testing MyDCGAN."""

    def __init__(self, dataloader, args, writer):
        """Initialize configurations."""

        self.dataloader = dataloader
        self.args = args
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""

        # Build Generator and Discriminator 
        self.G = Generator(self.args).to(self.device)
        self.D = Discriminator(self.args).to(self.device) 
        
        # Print networks information
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        # Apply custom weights
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        # Set criterion if adversarial_loss is chosen
        if(self.args.loss == 'adversarial_loss'):
            self.criterion = nn.BCELoss()
        
        # Set Adam optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.args.g_lr, [self.args.beta1, self.args.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.args.d_lr, [self.args.beta1, self.args.beta2])

    def print_network(self, model, name):
        """Print out the network information."""

        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("Number of parameters: {}".format(num_params))

    def load_model(self, resume_epochs, iter):
        """Restore the trained generator and discriminator."""
        
        G_path = os.path.join(self.args.model_save_dir, self.args.model_name,'iter_{}_epoch_{}-G.ckpt'.format(iter, resume_epochs))
        D_path = os.path.join(self.args.model_save_dir, self.args.model_name,'iter_{}_epoch_{}-D.ckpt'.format(iter, resume_epochs))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        if(self.args.mode == 'train'):
            print('Loading the trained models from iter [{}/{}] of epoch [{}/{}]...'.format(iter, len(self.dataloader), resume_epochs, self.args.epochs))
        else:
            print('Loading the trained model from epoch [{}] to create test images'.format(self.args.resume_epoch))

    def save_model (self,resume_epochs, iter):
        """Store the trained generator and discriminator."""

        G_path = os.path.join(self.args.model_save_dir, self.args.model_name, 'iter_{}_epoch_{}-G.ckpt'.format(iter, resume_epochs))
        D_path = os.path.join(self.args.model_save_dir, self.args.model_name, 'iter_{}_epoch_{}-D.ckpt'.format(iter, resume_epochs))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(self.args.model_save_dir))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
    
    def classification_loss(self, logit, target):
        """Compute binary cross entropy loss."""
        
        # Function that measures Binary Cross Entropy between target and input logits.
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def train(self):
        """Training function"""

        # Start training from scratch or resume training.
        start_epoch = 1
        start_iter = 1
        if self.args.resume_epoch and self.args.resume_iter:
            start_epoch = self.args.resume_epoch
            start_iter += self.args.resume_iter
            self.load_model(self.args.resume_epoch, self.args.resume_iter)

        # create a fixed_noise to generate example images for each selected attributes for debugging
        fixed_noise = torch.randn(5, self.args.nz, 1, 1, device=self.device)

        # Start training.
        print('Start training on device...', self.device)
        start_time = time.time()

        # Train could start from scratch or from given epoch and iteration
        for epoch in range (start_epoch, self.args.epochs + 1):

            print('-----------STARTING EPOCH [{}/{}]-----------'.format(epoch, self.args.epochs))

            for i, (images, labels) in enumerate(self.dataloader, 0):
                
                if(i == len(self.dataloader) - start_iter + 1):
                    start_iter = 1
                    break

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                self.D.zero_grad()
                b_size = images.size(0)
                label_org = labels.to(self.device)
                
                # Creating labels for generator and discriminator training with the right size
                broadcasted_labels = torch.zeros(b_size, self.args.label_dim, self.args.image_size, self.args.image_size, device = self.device)
                
                # generator input labels: (batch_size) x (label_dim) x (1) x (1) --> default: 128 x 3 x 1 x 1
                g_labels = labels.unsqueeze(-1).unsqueeze(-1).to(self.device)
                
                # discriminator input labels:  (batch_size) x (label_dim) x (image_size) x (image_size) --> default: 128 x 3 x 128 x 128
                d_labels = broadcasted_labels + g_labels
                
                if(self.args.loss == 'adversarial_loss'):
                    # Convention for real and fake labels during training with adversarial_loss
                    real_label = torch.full((b_size,), 1, dtype=torch.float, device=self.device)
                    fake_label = torch.full((b_size,), 0, dtype=torch.float, device=self.device)
                
                # Format batch
                real_images = images.to(self.device)

                
                # =================================================================================== #
                #                             2. Update D network                                     #
                # =================================================================================== #
                
                # --- Train D with all-real batch --- #
                # Forward pass real images through D
                d_output_real_src, d_output_real_cls  = self.D(real_images, d_labels, self.args.loss)
                d_output_real_src = d_output_real_src.view(-1)
                
                # Calculate loss on all-real images 
                errD_real = None
                if(self.args.loss == 'adversarial_loss'):
                    errD_real = self.criterion(d_output_real_src, real_label)
                elif(self.args.loss == 'wasserstein_loss'):
                    errD_real = - torch.mean(d_output_real_src)
                
                # Calculate classification_loss contribute if necessary
                d_loss_cls = 0
                if (self.args.classification_loss):
                    d_loss_cls = self.classification_loss(d_output_real_cls, label_org)
                
                # --- Train with all-fake images --- #
                # Generate noise input for generator 
                noise = torch.randn(b_size, self.args.nz, 1, 1, device = self.device)

                # Generate fake image batch with G with same labels as D
                fake = self.G(noise, g_labels)

                # Classify all fake images with D
                d_output_fake_src, d_output_fake_cls = self.D(fake.detach(), d_labels, self.args.loss)
                d_output_fake_src = d_output_fake_src.view(-1)

                totalD_loss = 0
                d_loss_gp = 0

                # Calculate loss on the all-fake images
                if(self.args.loss == 'adversarial_loss'):
                    errD_fake = self.criterion(d_output_fake_src, fake_label)
                elif (self.args.loss == 'wasserstein_loss'):
                    errD_fake = torch.mean(d_output_fake_src)
                    
                    # caluculate gradient penalty for wasserstein_loss
                    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device) # alpha*x + (1 - alpha)*x_2
                    x_hat = (alpha * real_images.data + (1 - alpha) * fake.data).requires_grad_(True)
                    output_src, output_cls = self.D(x_hat, d_labels, self.args.loss)
                    d_loss_gp = self.gradient_penalty(output_src, x_hat)
                
                # Calculate the gradients for this batch as sum of all contributes
                totalD_loss = errD_real + errD_fake + (self.args.lambda_gp * d_loss_gp) + (self.args.lambda_cls * d_loss_cls)
                totalD_loss.backward()
        
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                
                # Update D
                self.d_optimizer.step()

                # =================================================================================== #
                #                             3. Update G network                                     #
                # =================================================================================== #
                
                self.G.zero_grad()
                
                # Since we just updated D, perform another forward pass of all-fake images through D
                g_output_src, g_output_cls = self.D(fake, d_labels, self.args.loss)
                g_output_src = g_output_src.view(-1)

                # Compute classification loss contribute if necessary
                g_loss_cls = 0
                if (self.args.classification_loss):
                    g_loss_cls = self.classification_loss(g_output_cls, label_org)

                # Calculate G's loss based on this output
                if(self.args.loss == 'adversarial_loss'):
                    errG = self.criterion(g_output_src, real_label) 
                elif (self.args.loss == 'wasserstein_loss'):
                    errG = - torch.mean(g_output_src)
                
                # Calculate gradients for G
                errG += (self.args.lambda_cls * g_loss_cls)
                errG.backward()
                
                # Update G
                self.g_optimizer.step()

                # =================================================================================== #
                #                                          4. Miscellaneous                           #
                # =================================================================================== # 
                
                # Update loss items for tensorboard
                loss = {}
                loss['D/loss_real'] = errD_real.item()
                loss['D/loss_fake'] = errD_fake.item()
                if(self.args.classification_loss):
                    loss['D/loss_cls'] = d_loss_cls.item()
                    loss['G/loss_cls'] = g_loss_cls.item()
                loss['D/loss'] = errD.item()
                loss['G/loss'] = errG.item()
                if(self.args.loss == 'wasserstein_loss'):
                    loss['D/loss_gp'] = d_loss_gp.item()
                
                # Print out training information.
                if (i+1) % self.args.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}] on epoch [{}/{}]".format(et, i+start_iter, len(self.dataloader), epoch, self.args.epochs)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, (i+start_iter) + (epoch-1)*len(self.dataloader))

                # Check how the generator is doing by saving G's output on fixed_noise
                if ((i+1) % self.args.sample_step == 0):
                    with torch.no_grad():
                        fake = self.generate_test(fixed_noise, self.G).detach().cpu()
                    
                    # Split the result to make a grid of images
                    im_grid = vutils.make_grid(fake, padding=2, nrow = 5, normalize=True)
                    self.writer.add_image('samples_{}/_{}/_iter_{}_epoch_{}'.format(self.args.model_name, self.args.run_name, i+start_iter, epoch), im_grid, start_iter + 1)
                
                # Save model
                if(i+1) % self.args.model_save_step == 0:
                    self.save_model(epoch, i + start_iter)

            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            print('TRAINING TIME FOR EPOCH [{}/{}]: {}'.format(epoch, self.args.epochs, et))
            self.save_model(epoch, len(self.dataloader))
            start_iter = 1
        
        self.writer.flush()
        self.writer.close()

    def generate_test(self, fixed_noise, G):
        """Function to generate test samples. It will return a tensor composed by the concatenation
        of all the different images for all the different attributes. """
        
        # Switch to eval mode
        G.eval()

        # Create the conditioning tensor. It's an identity matrix of label_dim x label_dim size
        onehot = torch.eye(self.args.label_dim, self.args.label_dim)
        onehot = onehot.view(self.args.label_dim, self.args.label_dim, 1, 1)
        
        # If we want to try to apply all the attributes on the same image the conditioning
        # onehot vector is made of number ones only. In this case the number of different classes
        # of images will be 1. Otherwise it will be 'label_dim' i.e one for each attribute.
        classes = self.args.label_dim
        if self.args.attr_in_one_image:
            onehot = torch.ones(onehot.size()).to(self.device)
            classes = 1
        
        # We use the tensor c as index to get the onehot vector for each attribute.
        c = (torch.ones(5)*0).type(torch.LongTensor)
        c_onehot = onehot[c].to(self.device)

        # Create images from fixed noise, conditioned by the first element of onehot
        # the result will be 5 different images of a person with the first attribute
        out = G(fixed_noise, c_onehot)
        inference_res = out

        # Repeat the same operation for all the other attributes
        for l in range(1, classes):
            c = (torch.ones(5)*l).type(torch.LongTensor)
            c_onehot = onehot[c].to(self.device)
            out = G(fixed_noise, c_onehot)
            inference_res = torch.cat([inference_res, out], dim = 0)
        
        # Switch back to train mode
        G.train()

        return inference_res
    
    def test(self):
        """Testing function. It creates 'test_images' images for each attribute."""
        #manualSeed = 999

        # use if you want new results
        manualSeed = random.randint(1, 10000) 
        print("Random Seed: ", manualSeed)
        
        # Set seed
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        
        # Switch to eval mode
        self.G.eval()

        # Load the model from which to generate the test images
        self.load_model(self.args.resume_epoch, self.args.resume_iter)

        # Create the conditioning tensor
        onehot = torch.eye(self.args.label_dim, self.args.label_dim)
        onehot = onehot.view(self.args.label_dim, self.args.label_dim, 1, 1)
        
        # If we want to try to apply all the attributes on the same image the conditioning
        # onehot vector is made of number ones only. In this case the number of different classes
        # of images will be 1. Otherwise it will be 'label_dim' i.e one for each attribute.
        classes = self.args.label_dim
        if self.args.attr_in_one_image:
            onehot = torch.ones(onehot.size()).to(self.device)
            classes = 1

        inference_res = None
        with torch.no_grad():
            # Create 'test_images' different images (different input noise) for each attribute
            for i in range(self.args.test_images + 1):
                for l in range(0, classes):
                    noise = torch.randn(self.args.test_images, self.args.nz, 1, 1, device=self.device)
                    
                    # Use tensor c as index to get the onehot vector
                    c = (torch.ones(self.args.test_images)*l).type(torch.LongTensor)
                    c_onehot = onehot[c].to(self.device)
                    out = self.G(noise, c_onehot)
                    if l == 0:
                        inference_res = out
                    else:
                        inference_res = torch.cat([inference_res, out], dim = 0)
                im_grid = vutils.make_grid(inference_res.detach().cpu(), padding=2, nrow = self.args.test_images, normalize=True)
                self.writer.add_image('test_images_{}_epoch_{}'.format(self.args.model_name, self.args.resume_epoch), im_grid,  self.args.resume_epoch)
        print('test images created')
        return