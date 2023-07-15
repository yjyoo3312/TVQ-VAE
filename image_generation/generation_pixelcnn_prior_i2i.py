import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import TopicVectorQuantizedVAE_E2E
from tqdm import tqdm
from datasets import MiniImagenet

from tensorboardX import SummaryWriter

from functions import save_tensor_as_image

def main(args):     

    if args.dataset== 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train & test datasets
        train_dataset = datasets.CIFAR10(args.data_folder,
            train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(args.data_folder,
            train=False, transform=transform)
        valid_dataset = test_dataset
        num_channels = 3        
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True,
            download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True,
            download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True,
            download=True, transform=transform)
        num_channels = 3
    elif args.dataset == 'celeba':
        transform = transforms.Compose([
            transforms.CenterCrop((128, 128)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CelebA(args.data_folder,
                split='train', download=True, transform=transform)
        test_dataset = datasets.CelebA(args.data_folder,
                split='test', download=True, transform=transform)
        valid_dataset = datasets.CelebA(args.data_folder,
                split='valid', download=True, transform=transform)
        num_channels = 3
        
    # Define the data loaders
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=16, shuffle=True)
        
    
    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader)) 
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    save_tensor_as_image(fixed_grid, args.samples+"_original.png")    

    model = TopicVectorQuantizedVAE_E2E(num_channels, args.hidden_size_vae, args.k, args.n_clusters,
                                        args.hidden_size_prior, args.num_layers, fixed_images.shape[-1]//4).to(args.device)
    with open(args.vqvae_model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
        print(f"vq vae in {args.vqvae_model} loaded")
    model.eval()
         
    fixed_images = fixed_images.to('cuda')
    topics = model.get_topic_embedding_from_image(fixed_images)
    for idx, topic in tqdm(enumerate(topics)):
        topic = topic.unsqueeze(0).repeat(args.batch_size, 1)                
        latent = model.prior.generate_topic(label=topic, shape=(fixed_images.shape[-1]//4,fixed_images.shape[-1]//4), batch_size=args.batch_size)
        imgs = model.ae.decode(latent)
    
        imgs_grid = make_grid(imgs, nrow=8, range=(-1, 1), normalize=True)
        #writer.add_image(f'latent {str(0)}', imgs_grid, 0)
        save_tensor_as_image(imgs_grid, args.samples+"_"+str(idx)+".png")

    print('image generated')


    
if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='PixelCNN Prior for Topic VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str, default='data',
        help='name of the data folder')
    parser.add_argument('--dataset', type=str, default='cifar10',
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--vqvae_model', type=str, default='models/tvqvae_cifar10_e2e_10/best_loss_prior.pt',
        help='filename containing the vqvae model')
    parser.add_argument('--samples', type=str, default='samples/cifar10_pixel_prior_e2e',
        help='filename containing the prior model')
    parser.add_argument('--n_clusters', type=int, default=10,
        help='topic number(default: 20)')
    parser.add_argument('--batch_size', type=int, default=64,
        help='batch_size')

    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--hidden-size-prior', type=int, default=64,
        help='hidden size for the PixelCNN prior (default: 64)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')
    parser.add_argument('--num-layers', type=int, default=15,
        help='number of layers for the PixelCNN prior (default: 15)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='samples',
        help='name of the output folder (default: prior)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./samples'):
        os.makedirs('./samples')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')    
    args.steps = 0

    main(args)
