from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import numpy as np
from collections import OrderedDict

def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
 
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)       
        self._embedding.weight.data.uniform_(-1, 1)
        self._commitment_cost = commitment_cost
        self._embedding_cat = None
                   
    def forward(self, inputs, n=1):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
       
        embedding = self._embedding
        num_embeddings = self._num_embeddings
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, embedding.weight.t()))
        
        # Encoding
        distance_indices, encoding_indices = torch.topk(-1*distances, k=n, dim=1)
        distance_indices = F.normalize(1/(-1*distance_indices), dim=-1)
        encodings = torch.zeros(encoding_indices.shape[0], num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, distance_indices.detach())
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
    
class AutoEncoderVQ(nn.Module):

    def __init__(self, input_dim, num_embedding, hidden_dims, dropout, n):
        super(AutoEncoderVQ, self).__init__()

        self.num_embedding = num_embedding
        self.encoder_layers = []
        self.n = n
        dims = [input_dim] + hidden_dims
        self.vq_vae = VectorQuantizer(num_embeddings=num_embedding, embedding_dim=hidden_dims[-1])
        self.z_drop = nn.Dropout(dropout)

        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.encoder_layers.append(layer)
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        self.decoder_layers = []
        hidden_dims.reverse()
        dims = hidden_dims + [input_dim]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.decoder_layers.append(layer)
        self.decoder = nn.Sequential(*self.decoder_layers)
        
    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=-1)

        z = self.z_drop(z)
        z = self.pre_vq(z)
        loss_vq, quantized, perplexity, encodings = self.vq_vae(z, n=self.n)

        z_q = self.post_vq(quantized)        
        x_bar = self.decoder(z_q)

        return loss_vq, perplexity, x_bar, z, z_q, encodings
    
    def pre_vq(self, x):
        return x.unsqueeze(2).unsqueeze(3) #convert (B,D) to (B,D,1,1)
    
    def post_vq(self,x):
        return x.squeeze()
    
class InferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size, output_size, hidden_sizes=(100, 100), dropout=0.0):
        
        super(InferenceNetwork, self).__init__()        

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.activation = nn.Tanh()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)    

    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma
   

class DecoderNetwork(nn.Module):

    """AVITM Network."""

    def __init__(self, input_size, word_size, n_components=10, hidden_sizes=(100, 100), dropout=0.0):

        super(DecoderNetwork, self).__init__()       

        self.input_size = input_size
        self.word_size = word_size
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        
        topic_prior_mean = 1 / n_components

        self.inf_net = InferenceNetwork(input_size, n_components, hidden_sizes, dropout=dropout)
        self.prior_mean = nn.Parameter(torch.tensor([topic_prior_mean] * n_components))
            
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = nn.Parameter(torch.tensor([topic_prior_variance] * n_components))            

        self.beta = nn.Parameter(torch.Tensor(n_components, input_size))        
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        nn.init.xavier_uniform_(self.beta)
       
        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)    

    def forward(self, x):        

        posterior_mu, posterior_log_sigma = self.inf_net(x)
        posterior_sigma = torch.exp(posterior_log_sigma)

        theta = F.softmax(reparameterize(posterior_mu, posterior_log_sigma), dim=1)           
        topic_doc = theta
        theta = self.drop_theta(theta)

        word_dist = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)          
        topic_word = self.beta
        self.topic_word_matrix = self.beta
        
        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, topic_word,topic_doc

    def get_theta(self, x):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x)

        # generate samples from theta
        return F.softmax(reparameterize(posterior_mu, posterior_log_sigma), dim=1)

    
class TopicVectorQuantizedVAE(BertPreTrainedModel):

    def __init__(self, config, input_dim, hidden_dims, n_embeddings, n_words, n_clusters, n, dropout=0.0, alpha_hidden=1): 
                 
        super().__init__(config)
        self.init_weights()        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.ae = AutoEncoderVQ(input_dim, n_embeddings, hidden_dims, dropout, n)
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.dropout = dropout              

        self.topic_model_bow = DecoderNetwork(n_embeddings, n_words, n_clusters, dropout=dropout)     

        if alpha_hidden == 1:
            hidden = (100,100)
        elif alpha_hidden == 2:
            hidden = (100,100,100)
        elif alpha_hidden == 3:
            hidden = (100, 100, 100, 100)
        else:
            hidden = (100,100)   

        # link embedding to topic       
        self.alpha = InferenceNetwork(input_size=hidden_dims[0], hidden_sizes=hidden, output_size=n_words, dropout=dropout)
        #self.alpha = nn.Parameter(torch.Tensor(hidden_dims[0], n_words))
        self.alpha_bn = nn.BatchNorm1d(n_words, affine=False)
        self.emb = self.ae.vq_vae._embedding.weight
        
        # topic word, topic dist
        self.topic_word = None
        self.topic_document = None
       
        for param in self.bert.parameters():
            param.requires_grad = False

    def _loss_bow(self, inputs, inputs_o, word_dists, word_dists_o, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)

        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)

        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)

        KL = 0.5 * (var_division + diff_term - self.n_clusters + logvar_det_division)
        KL = KL.mean()

        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        RL = RL.mean()

        # Reconstruction term for original word dist
        RL_o = None
        if word_dists_o is not None:
            RL_o = -torch.sum(inputs_o * torch.log(word_dists_o + 1e-10), dim=1)
            RL_o = RL_o.mean()

        return KL, RL, RL_o
    
  
    def disable_ae_grads(self):
        for param in self.ae.parameters():
            param.requires_grad = False

    def enable_ae_grads(self):
        for param in self.ae.parameters():
            param.requires_grad = True
        
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
    
        
    def forward_topic(self, bows, expansion=True):

        prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
            word_dists, topic_words, topic_document = self.topic_model_bow(bows)
        
        theta = topic_document
        beta = topic_words
        word_dists_o = None

        if expansion:
            emb = F.normalize(self.emb, p=2, dim=1)
            beta_n = F.normalize(torch.matmul(beta,emb), p=2, dim=1)
            #beta_o = torch.matmul(beta_n, self.alpha)
            beta_o, _ = self.alpha(beta_n)           
            word_dists_o = F.softmax(self.alpha_bn(torch.matmul(theta, beta_o)), dim=1)

            self.topic_document = theta
            self.topic_word = beta_o

        
        return prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
            word_dists, word_dists_o, topic_words, topic_document
        
   
    def forward(self, input_ids, attention_mask, valid_pos):
        self.bert.eval()
        bert_outputs = self.bert(input_ids,attention_mask=attention_mask)                                 
        last_hidden_states = bert_outputs[0]
        attn_mask = valid_pos != 0
        input_embs = last_hidden_states[attn_mask]   
                  
        loss_vq, perplexity, output_embs, z, _, encodings = self.ae(input_embs)

        return input_embs, output_embs, z, encodings, attn_mask, loss_vq, perplexity


    

