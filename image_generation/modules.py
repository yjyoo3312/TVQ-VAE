import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from collections import OrderedDict
import numpy as np

from functions import vq, vq_st

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            4*dim, 2 * dim
        )
        #self.class_cond_embedding = nn.Embedding(
        #   n_classes, 2 * dim
        #)

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = torch.matmul(h, self.class_cond_embedding.weight)
        #h = self.class_cond_embedding(h)
        
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        label = label.expand(batch_size)

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x
    
    def generate_topic(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )        

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x

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

        #self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_layer = nn.Conv2d(input_size, hidden_sizes[0], kernel_size=kernel_size, stride=1, padding=0)


        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = self.input_layer(x).squeeze()
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma
   

class DecoderNetwork(nn.Module):
    def __init__(self, input_size, n_components=10, model_type='prodLDA', kernel_size=8,
                 hidden_sizes=(100,100), activation='tanh', dropout=0.2,
                 topic_prior_mean=0.0):
        
        super(DecoderNetwork, self).__init__()       

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout

        self.inf_net = InferenceNetwork(input_size, n_components, hidden_sizes, activation, kernel_size=kernel_size)
        #self.inf_net = InferenceNetwork(input_size, n_components, hidden_sizes, activation)

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
        #concat
        #x = torch.cat((x, x_o), dim=1)
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
    

        
class TopicVectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_embeddings, n_clusters=10, n=1, 
                 t_hidden_size=300, dropout=0.2, model_type='prodLDA'):
        super().__init__()
 
        self.ae = VectorQuantizedVAE(input_dim, hidden_dims, n_embeddings)
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.dropout = dropout      
        self.t_hidden_size = t_hidden_size
        self.n_embeddings = n_embeddings

        # get prodLDA
        self.topic_model_bow = DecoderNetwork(n_embeddings, n_clusters, model_type, dropout=dropout)        

        # link embedding to topic       
        
        # topic word, topic dist
        self.topic_word = None
        self.topic_document = None

    def disable_ae_grads(self):
        for param in self.ae.parameters():
            param.requires_grad = False

    def enable_ae_grads(self):
        for param in self.ae.parameters():
            param.requires_grad = True

    def load_ae_params(self, path):
        with open(path, 'rb') as f:
            state_dict = torch.load(f)
            self.ae.load_state_dict(state_dict)
            print(f"vq vae in {path} loaded")

    def get_topic_embedding(self, x):
        theta = self.topic_model_bow.get_theta(x)                   
        return torch.matmul(torch.matmul(theta, self.topic_model_bow.beta), self.ae.codebook.embedding.weight)
    
      
    
    def get_embedding_per_topic(self, topic_index, batch_size=64):
        topic_onehot = F.one_hot(topic_index, self.n_clusters).float()
        theta = topic_onehot.unsqueeze(0).repeat(batch_size, 1)
        emb = torch.matmul(theta, self.topic_model_bow.beta)

        return torch.matmul(emb, self.ae.codebook.embedding.weight)


    def _loss_bow(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        # var division term

        var_division = torch.sum(posterior_variance / (prior_variance), dim=1)
        
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        
        # combine terms
        KL = 0.5 * (var_division + diff_term - self.n_clusters + logvar_det_division)
        KL = KL.sum()

        # Reconstruction term
        bows = inputs.sum(dim=(2,3))
        RL = -torch.sum(bows * torch.log(word_dists + 1e-10), dim=1)
        RL = RL.sum()
       

        return KL, RL
        
    def get_info(self, vocab, topk=25):
        info = {}                   

        info['topic-word-matrix'] = self.topic_word.cpu().detach().numpy()
        info['topic-document-matrix'] = self.topic_document.cpu().detach().numpy().T

        topic_w = []
        for k in range(self.n_clusters):
                if np.isnan(info['topic-word-matrix'][k]).any():
                    # to deal with nan matrices
                    topic_w = None
                    break
                else:
                    top_words = list(
                        info['topic-word-matrix'][k].argsort()[-topk:][::-1])
                topic_words = [vocab[a] for a in top_words]
                topic_w.append(topic_words)
        info['topics'] = topic_w  

        return info                    
    
        
    def forward_topic(self, x):

        bows = F.one_hot(self.ae.encode(x), self.n_embeddings).permute(0,3,1,2).float()

        prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
            word_dists, topic_words, topic_document = self.topic_model_bow(bows)
               

        kld_theta, recon_loss_e = self._loss_bow(bows, word_dists, prior_mean, prior_var, 
                                                 posterior_mean, posterior_var, posterior_log_var)
                                                
                                                                             
        
        return kld_theta, recon_loss_e, topic_words, topic_document
    


class TopicVectorQuantizedVAE_E2E(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_embeddings, n_clusters=10, n_hidden_size_prior=64, num_prior_layers=15, kernel_size=8,
                 t_hidden_size=300, dropout=0.0, model_type='prodLDA'):
        super().__init__()
 
        self.ae = VectorQuantizedVAE(input_dim, hidden_dims, n_embeddings)
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.dropout = dropout      
        self.t_hidden_size = t_hidden_size
        self.n_embeddings = n_embeddings

        # get prodLDA
        self.topic_model_bow = DecoderNetwork(n_embeddings, n_clusters, model_type, dropout=dropout, kernel_size=kernel_size)    

        # get autoregressive generation prior
        self.prior = GatedPixelCNN(n_embeddings, n_hidden_size_prior, num_prior_layers)
        
        # topic word, topic dist
        self.topic_word = None
        self.topic_document = None

    def disable_ae_grads(self):
        for param in self.ae.parameters():
            param.requires_grad = False

    def enable_ae_grads(self):
        for param in self.ae.parameters():
            param.requires_grad = True

    def load_ae_params(self, path):
        with open(path, 'rb') as f:
            state_dict = torch.load(f)
            self.ae.load_state_dict(state_dict)
            print(f"vq vae in {path} loaded")

    def get_topic_embedding(self, x):
        theta = self.topic_model_bow.get_theta(x)                   
        return torch.matmul(torch.matmul(theta, self.topic_model_bow.beta), self.ae.codebook.embedding.weight)
    
    def get_topic_embedding_2(self, x):
        theta = self.topic_model_bow.get_theta(x)                 
        return torch.matmul(torch.matmul(theta, self.topic_model_bow.beta), self.ae.codebook.embedding.weight), theta
    
    def get_topic_embedding_from_image(self, x):
        return self.get_topic_embedding(self.ae.encode(x))

    def get_topic_embedding_from_image_2(self, x):
        embedding, theta = self.get_topic_embedding_2(self.ae.encode(x))
        return embedding, theta
    
    
    def get_embedding_per_topic(self, topic_index, batch_size=64):
        topic_onehot = F.one_hot(topic_index, self.n_clusters).float()
        theta = topic_onehot.unsqueeze(0).repeat(batch_size, 1)
        emb = torch.matmul(theta, self.topic_model_bow.beta)

        return torch.matmul(emb, self.ae.codebook.embedding.weight)


    def _loss_bow(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):

        var_division = torch.sum(posterior_variance / (prior_variance), dim=1)

        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        
        KL = 0.5 * (var_division + diff_term - self.n_clusters + logvar_det_division)
        KL = KL.mean()

        bows = inputs.sum(dim=(2,3))
        RL = -torch.sum(bows * torch.log(word_dists + 1e-10), dim=1)
        RL = RL.mean()
       

        return KL, RL

    def _loss_prior(self, x):
        latents = self.ae.encode(x)
        latents = latents.detach()
        topics = self.get_topic_embedding(latents)
        logits = self.prior(latents, topics)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        loss = F.cross_entropy(logits.view(-1, self.n_embeddings), latents.view(-1))                                   
                
        return loss
        
    def get_info(self, vocab, topk=25):
        info = {}                   

        info['topic-word-matrix'] = self.topic_word.cpu().detach().numpy()
        info['topic-document-matrix'] = self.topic_document.cpu().detach().numpy().T

        topic_w = []
        for k in range(self.n_clusters):
                if np.isnan(info['topic-word-matrix'][k]).any():
                    # to deal with nan matrices
                    topic_w = None
                    break
                else:
                    top_words = list(
                        info['topic-word-matrix'][k].argsort()[-topk:][::-1])
                topic_words = [vocab[a] for a in top_words]
                topic_w.append(topic_words)
        info['topics'] = topic_w  

        return info                    
    
        
    def forward_topic(self, x):

        bows = F.one_hot(self.ae.encode(x), self.n_embeddings).permute(0,3,1,2).float()

        prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
            word_dists, topic_words, topic_document = self.topic_model_bow(bows)
               

        kld_theta, recon_loss_e = self._loss_bow(bows, word_dists, prior_mean, prior_var, 
                                                 posterior_mean, posterior_var, posterior_log_var)
        
        return kld_theta, recon_loss_e, topic_words, topic_document
                                                
    