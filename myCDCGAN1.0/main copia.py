import os
import argparse
from solver import Solver
from data_loader import get_loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter


def str2bool(v):
    return v.lower() in ('true')

def get_args():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')

    parser.add_argument('--label_dim', type=int, default=3, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--loss', type=str, default='adversarial_loss', choices=['adversarial_loss', 'wasserstein_loss'], help='type of loss to use')

    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    # parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    # parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    # parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    # parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--nz', type=int, default=100, help='size of z latent vector (i.e. size of generator input)')
    parser.add_argument('--ngf', type=int, default=64, help='size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='size of feature maps in discriminator')
    parser.add_argument('--nc', type=int, default=3, help='Number of channels in the training images.')




    #parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    #parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    #parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    # parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    # parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this epoch')
    parser.add_argument('--resume_iter', type=int, default=None, help='resume training from this iteration')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset', default=['Black_Hair', 'Male', 'Young'])

    # Test configuration.
    #parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_samples', type=int, default=None, help='Samples to use during training')

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--log_dir', type=str, default='mydcgan/logs')
    parser.add_argument('--model_save_dir', type=str, default='mydcgan/models')
    parser.add_argument('--sample_dir', type=str, default='mydcgan/samples')
    parser.add_argument('--result_dir', type=str, default='mydcgan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=200)
    #parser.add_argument('--lr_update_step', type=int, default=1000)

    args = parser.parse_args()

    if(args.label_dim != len(args.selected_attrs)):
        raise Exception("Label dimension and number of selected attributed are different.")
    
    return args


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def main(args):
    # For fast training.
    cudnn.benchmark = True
    writer = SummaryWriter('./runs/' + args.run_name)
    path = args.model_save_dir + '/' + args.model_name
    # Create directories if not exist.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Data loader.
    celeba_loader = get_loader(args.celeba_image_dir, args.attr_path, args.selected_attrs,
                                   args.celeba_crop_size, args.image_size, args.batch_size,
                                   args.mode, args.num_workers, args.num_samples)




    #Some example image from Data Loader
    # to_pil = transforms.ToPILImage()

    # fixed_batch, fixed_c = next(iter(celeba_loader))

    # print(args.selected_attrs)

    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     figure.add_subplot(rows, cols, i)
    #     plt.axis("off")
    #     plt.title(fixed_c[i])
    #     plt.imshow(to_pil(denorm(fixed_batch[i])))
    # plt.show()
    

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, args, writer)

    if args.mode == 'train':
            solver.train()
    elif args.mode == 'test':
            solver.test()


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)