import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from collections import OrderedDict
import numpy as np

class InferenceNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes,
                 activation='tanh', dropout=0.2, kernel_size=16):
        
        super(InferenceNetwork, self).__init__()        

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'rrelu':
            self.activation = nn.RReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        #self.input_layer = nn.Conv2d(input_size, hidden_sizes[0], kernel_size=kernel_size, stride=1, padding=0)


        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma
   

class DecoderNetwork(nn.Module):
    def __init__(self, input_size, n_components=10, kernel_size=8,
                 hidden_sizes=(512,256,128,128,256,512), activation='tanh', dropout=0.2,
                 topic_prior_mean=0.0, model_type='prodLDA'):
        
        super(DecoderNetwork, self).__init__()       

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout

        self.inf_net = InferenceNetwork(input_size, n_components, hidden_sizes, activation, kernel_size=kernel_size)        

        #self.topic_prior_mean = topic_prior_mean
        self.prior_mean = nn.Parameter(torch.tensor([topic_prior_mean] * n_components))
            
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = nn.Parameter(torch.tensor([topic_prior_variance] * n_components))            

        self.beta = nn.Parameter(torch.Tensor(n_components, input_size))        
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        nn.init.xavier_uniform_(self.beta)
       
        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):

        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        topic_doc = theta
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            topic_word = self.beta
            # word_dist: batch_size x input_size
        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            topic_word = beta
            word_dist = torch.matmul(theta, beta)

            # word_dist: batch_size x input_size

        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, topic_word,topic_doc

    def get_theta(self, x):
        bows = F.one_hot(x, self.input_size).permute(0,3,1,2).float()
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(bows)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta