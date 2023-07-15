import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import TopicVectorQuantizedVAE_E2E
from tqdm import tqdm

from tensorboardX import SummaryWriter

from functions import save_tensor_as_image

def main(args):      
    
    num_channels = 3 

    model = TopicVectorQuantizedVAE_E2E(num_channels, args.hidden_size_vae, args.k, args.n_clusters,
                                        args.hidden_size_prior, args.num_layers, 8).to(args.device)
    with open(args.vqvae_model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
        print(f"vq vae in {args.vqvae_model} loaded")
    model.eval()
         

    for idx in tqdm(range(args.n_clusters)):
        topics = model.get_embedding_per_topic(topic_index=torch.tensor(idx).to(args.device), batch_size=64)
        latent = model.prior.generate_topic(label=topics, shape=(8,8), batch_size=64)
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

    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--hidden-size-prior', type=int, default=64,
        help='hidden size for the PixelCNN prior (default: 64)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')
    parser.add_argument('--num-layers', type=int, default=15,
        help='number of layers for the PixelCNN prior (default: 15)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=200,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-5,
        help='learning rate for Adam optimizer (default: 3e-4)')

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
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
