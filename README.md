# MyCDCGAN
### Project objective
Given celeba dataset and the corresponding image attributes create a cGAN that can generate images belonging to a specific class (For example a human face with blond hair or smiling)
### Dataset
CelebA dataset
### Network model
- A conditional GAN is used for performing this task. 
- The dataloader returns an image and its corresponding attribute labels (StarGAN loader was used as a starting point: https://github.com/yunjey/stargan
- The number of attributes to generate can be decided a-priori. For example you can select 5 attributes and train the model on those.

# Python Modules
- torch
- torchvision
- matplotlib
- PIL
- os
- random
- numpy
- time
- datetime
- tensorboard

# Download and set up
To download the network code:
```
git clone https://github.com/giorgiatedaldi/MyCDCGAN
cd MyCDCGAN/
```
To download CelebA dataset use the download.sh file from https://github.com/yunjey/stargan.
When download is complete copy the **data** folder in **MyCDCGAN** folder. (This is the default path to CelebA dataset).

# Network train
Each parameter is configurable from command line:
```
    parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')
    parser.add_argument('--label_dim', type=int, default=3, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--loss', type=str, default='adversarial_loss', choices=['adversarial_loss', 'wasserstein_loss'], help='type of loss to use')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--nz', type=int, default=100, help='size of z latent vector (i.e. size of generator input)')
    parser.add_argument('--ngf', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--ndf', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--nc', type=int, default=3, help='number of channels in the training images.')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--classification_loss', type=int, default=0, help='enable (1) or disable (0) classification loss')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this epoch')
    parser.add_argument('--resume_iter', type=int, default=None, help='resume training from this iteration')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset', default=['Wearing_Hat', 'Male', 'Eyeglasses'])

    # Test configuration.
    parser.add_argument('--test_images', type=int, default=2, help='test images to generate for each attribute')
    parser.add_argument('--attr_in_one_image', type=int, default=0, help='all selected attributes will be applied to the same image')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1, help='num workers to use')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test the model')
    parser.add_argument('--num_samples', type=int, default=None, help='samples to use during training')

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images', help='specify the path to celeba "images" folder')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt', help='specify the path to celeba "list_attr_celeba.txt" file')
    parser.add_argument('--model_save_dir', type=str, default='mydcgan/models', help = 'specify path to folder where to save model checkpoints')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10, help = 'log information every "n" iterations')
    parser.add_argument('--sample_step', type=int, default=100, help = 'save images on tensorboard every "n" iterations')
    parser.add_argument('--model_save_step', type=int, default=200, help = 'save model every "n" iterations')
```

Example to start a train from scratch:
```
python3 main.py --model_name wass_classification_model --label_dim 5 --selected_attr Wearing_Hat Eyeglasses Bangs Blond_Hair Wavy_Hair --batch_size 64 --loss wasserstein_loss --classification_loss 1 --run_name wass_class 
```
Example to start a train from checkpoint:
```
python3 main.py --model_name wass_classification_model --label_dim 5 --selected_attr Wearing_Hat Eyeglasses Bangs Blond_Hair Wavy_Hair --batch_size 64 --loss wasserstein_loss --classification_loss 1 --run_name wass_class --resume_epoch 5 --resume_iter 3135
```

# Network test
```
python3 main.py --model_name wass_classification_model --label_dim 5 --selected_attr Wearing_Hat Eyeglasses Bangs Blond_Hair Wavy_Hair --batch_size 64 --run_name test --resume_epoch 5 --resume_iter 3135 --mode test --test_images 6
```

# Result
Some results are store in the **runs** folder. They can be seen trough **tensorboard**. From the folder were **runs** is saved you can run the following command:
```
tensorboard --logdir=runs
```

# Checkpoints
Some checkpoints of different trainings are available in the **mydcgan/models** folder.
