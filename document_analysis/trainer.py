import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from model import TopicVectorQuantizedVAE
import os
import argparse
import scipy.sparse as sp
from utils import TVQVAEClusUtils, AverageMeter
import numpy as np
import nltk
import pickle
import math
import warnings
from tqdm import tqdm
from collections import OrderedDict
warnings.filterwarnings("ignore")

class TVQVAETrainer(object):

    def __init__(self, args):
        if args is not None:
            self.args = args        

        self.args = args
        pretrained_lm = 'bert-base-uncased'
        self.n_clusters = args.n_clusters
        self.n_embeddings = args.n_embeddings
        self.batch_size = args.batch_size
        self.n = args.n
        self.k = args.k
        self.latent_dim = eval(args.hidden_dims)[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.utils = TVQVAEClusUtils()
       
        #setting pathes
        result_base = args.result_base+"_"+args.dataset
        self.data_dir = os.path.join(args.dataset_base, args.dataset)
        self.res_dir = f"results_{args.dataset}_{args.seed}_{args.n_clusters}_{args.n_embeddings}_{args.n}"
        self.res_dir = os.path.join(result_base, self.args.model_selection, self.res_dir)
        self.label_dir = os.path.join(self.data_dir, args.label_path)        
        
        if args.dataset == '20ng':
            self.split = [11415, 13862] 
        elif args.dataset == 'nyt': #following the settings form ETM / 85% 10% 5%
            self.split = [27197, 30397]     
        elif args.dataset == 'yelp':
            self.split = [24888,27816]
        elif args.dataset == 'm10':
            self.split = [7509,8393]
        elif args.dataset == 'dblp':
            self.split = [46405,51865]
        elif args.dataset == 'imdb':
            self.split = [42500,47500]     

        #dataset setting
        tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
        self.vocab_from_bert = tokenizer.get_vocab()
        self.inv_vocab_from_bert = {k:v for v, k in self.vocab_from_bert.items()}
        
        self.utils.get_bert_vocabs(self.vocab_from_bert, self.inv_vocab_from_bert, tokenizer)               
                                   
        self.utils.create_dataset(self.data_dir, "corpus.txt", "text.pt")
        data, data_test, _ = self.utils.load_dataset(self.data_dir, "text.pt", split=self.split)

        #Define model
        self.n_words = max(self.utils.vocab_bert2gensim.values())+1                       
        self.model = TopicVectorQuantizedVAE.from_pretrained(pretrained_lm,
                                                            output_attentions=False,
                                                            output_hidden_states=False,
                                                            input_dim=args.input_dim,
                                                            hidden_dims=eval(args.hidden_dims),
                                                            n_embeddings=args.n_embeddings,
                                                            n_words=self.n_words,
                                                            n_clusters=args.n_clusters,
                                                            n= args.n,
                                                            alpha_hidden=args.alpha_hidden)
                                                               
        
        #arrays for topic calculation
        self.td,self.tw = None, None

        input_ids = data["input_ids"]
        attention_masks = data["attention_masks"]
        valid_pos = data["valid_pos"]
        doc_id = torch.arange(len(input_ids))

        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.valid_pos = valid_pos
        self.doc_id = doc_id

        input_ids_test = data_test["input_ids"]
        attention_masks_test = data_test["attention_masks"]
        valid_pos_test = data_test["valid_pos"]
        doc_id_test = torch.arange(len(input_ids_test))
        
        self.data = TensorDataset(input_ids, attention_masks, valid_pos, doc_id)
        self.data_test = TensorDataset(input_ids_test, attention_masks_test, valid_pos_test, doc_id_test)
                                           
        os.makedirs(self.res_dir, exist_ok=True)
        self.log_files = {}    

    def load_pretrained_path(self, pretrain=False):
        test_pretrained_path = "pretrained_vq_"+str(self.args.n_embeddings)+"_"+str(self.args.n)
        if pretrain:
            test_save_path = "pretrained_vq_"+str(self.args.n_embeddings)+"_"+str(self.args.n)            
        else:
            test_save_path = "pretrained_vq_"+str(self.args.n_clusters)+"_"+str(self.args.n_embeddings)+"_"+str(self.args.n)
            
        
        pretrained_vq_path = os.path.join(self.data_dir, self.args.model_selection, test_pretrained_path)
        save_vq_path = os.path.join(self.data_dir, self.args.model_selection, test_save_path)

        return pretrained_vq_path, save_vq_path
    
        
    def initialize_vq(self, pretrained_vq_path, save_vq_path, epochs=500):      
        pt_path = os.path.join(pretrained_vq_path, 'best.pt')
        print(pt_path)

        if os.path.exists(pt_path):
            pretrained_ae_params = torch.load(pt_path)
            pretrained_ae_params = torch.load(pt_path)
            ae_params = OrderedDict([(key[3:], value) for key, value in pretrained_ae_params.items() if key.startswith('ae.')])
            self.model.ae.load_state_dict(ae_params)            
            print(f'Loaded pretrained state dicts from {pt_path}')
        else:
            print('no pretrained ae models. running from scratch')  

        self.model.to(self.device)
       
        data = TensorDataset(self.input_ids, self.attention_masks, self.valid_pos, self.doc_id)
        sampler = SequentialSampler(data)
        dataset_loader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        doc_encs = []
        o_doc_encs = []
        with torch.no_grad():
            for batch in tqdm(dataset_loader, total=len(dataset_loader), desc="stacking encodings"):
                input_ids, attention_mask, valid_pos, max_len = self.utils.get_bert_input(batch[:3], device=self.device)
                doc_ids = batch[3].to(self.device)
                _, _, _, w_enc, attn_mask, _, _ = self.model(input_ids, attention_mask, valid_pos)  
                doc_enc, o_doc_enc, _ = self.utils.calc_doc_enc(w_enc, input_ids = input_ids, doc_ids=doc_ids, attn_mask=attn_mask, max_len=max_len)
                doc_encs.append(doc_enc)
                o_doc_encs.append(o_doc_enc)

        doc_encs = torch.cat(doc_encs, dim=0)
        o_doc_encs = torch.cat(o_doc_encs, dim=0)
        data_doc = TensorDataset(doc_encs, o_doc_encs)
        sampler = RandomSampler(doc_encs)
        dataset_loader = DataLoader(data_doc, sampler=sampler, batch_size=self.batch_size)

        self.model.disable_ae_grads()
        optimizer_tp = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr) #bert not trained.
        self.model.train()
        self.model.ae.eval()
       
        print('pretrain td')
        for epoch in range(epochs):

            total_loss_rec_o, total_loss_rec_e, total_loss_kld = AverageMeter(), AverageMeter(), AverageMeter()

            for batch in dataset_loader:
                optimizer_tp.zero_grad()               
                doc = batch[0].to(self.device)
                o_doc = batch[1].to(self.device)

                prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
                word_dists, word_dists_o, _, _ = self.model.forward_topic(doc, expansion=True)

                kld_theta, recon_loss_e, recon_loss_o = self.model._loss_bow(doc, o_doc, word_dists, word_dists_o, prior_mean, prior_var,
                                                                             posterior_mean, posterior_var, posterior_log_var)       

                #loss = recon_loss_o + recon_loss_e + kld_theta
                loss = recon_loss_o + kld_theta

                loss.backward()               
                optimizer_tp.step()

                total_loss_rec_o.update(recon_loss_o.item())
                total_loss_rec_e.update(recon_loss_e.item())
                total_loss_kld.update(kld_theta.item())

            if epoch % int(epochs//5) == 0 and epoch > 0:
                print(f"Epoch {epoch} loss recon_e = {total_loss_rec_e.avg:.4f} loss recon_o = {total_loss_rec_o.avg:.4f} loss kld = {total_loss_kld.avg:.4f}")

        npmi, diversity, _,_, perplexity = self.inference(epoch, suffix="_initial")
        tq = npmi*diversity # topic quality                
        print(f"Initial npmi={npmi:.4f}: diversity={diversity:.4f}: tq={tq:.4f}") 
        torch.save(self.model.state_dict(), os.path.join(save_vq_path, "init.pt"))

        self.model.enable_ae_grads()

    
    def pretrain(self, train_epoch=20):
        pretrained_vq_path, _ = self.load_pretrained_path(pretrain=True)

        os.makedirs(pretrained_vq_path, exist_ok=True)
        print(f"Pretraining VQ-VAE")
        sampler = RandomSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
        model = self.model.to(self.device)
        model.eval()            

        iter_num = len(dataset_loader)*train_epoch
        print(f'Number of total pretrain iteration: {iter_num}')
                                                         
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.ae.parameters()), lr=self.args.lr)        
        
        #pretraining vq-vae
        for epoch in range(train_epoch):
            total_loss, total_loss_mse, total_loss_vq, total_perplexity = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

            model.ae.train()
            for batch_idx, batch in enumerate(dataset_loader):                
                optimizer.zero_grad()
                input_ids, attention_mask, valid_pos, __build_class__ = self.utils.get_bert_input(batch[:3], device=self.device)
                input_embs, output_embs, _, _, _, loss_vq, perplexity = self.model(input_ids, attention_mask, valid_pos) 

                loss_mse = F.mse_loss(output_embs, input_embs) 
                loss = loss_mse + loss_vq

                total_loss.update(loss.item())
                total_loss_mse.update(loss_mse.item())
                total_loss_vq.update(loss_vq.item())
                total_perplexity.update(perplexity.item())

                loss.backward()
                optimizer.step()
            
                if batch_idx % int(len(dataset_loader)//4) == 0 and batch_idx > 0:                      
                    print(f"epoch {epoch}: idx {batch_idx}: loss_mse = {total_loss_mse.avg:.4f}: "
                          +f"loss_vq {total_loss_vq.avg:.7f}")
                    
        print('saving pretrained files')
        torch.save(self.model.state_dict(), os.path.join(pretrained_vq_path, "best.pt"))

       
    # train autoencoder with reconstruction loss
    def train(self, train_epoch=20):                   
        
        pretrained_vq_path, save_vq_path = self.load_pretrained_path()        
        
        print(f'load {pretrained_vq_path}')
        print(f'save to {save_vq_path}')
        os.makedirs(pretrained_vq_path, exist_ok=True)    
        os.makedirs(save_vq_path, exist_ok=True) 
        
        self.initialize_vq(pretrained_vq_path, save_vq_path)
        self.model.to(self.device)
                 
        self.data = TensorDataset(self.input_ids, self.attention_masks, self.valid_pos, self.doc_id)
        sampler = RandomSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size, shuffle=False) 

        iter_num = len(dataset_loader)*train_epoch
        print(f'Number of total iteration: {iter_num}') 
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr) 

        nmi_best, npmi_best, tq_best =-1.0, -1.0, -1.0
        purity_best, diversity_best = -1.0, -1.0
        best_epoch = 0  

        #pretraining vq-vae
        for epoch in range(train_epoch): 

            total_loss_rec_e, total_loss_rec_o, total_loss_kld, total_loss_ae  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

            self.model.train()
            self.model.bert.eval()
            for batch_idx, batch in enumerate(dataset_loader):                
                optimizer.zero_grad()

                input_ids, attention_mask, valid_pos, max_len = self.utils.get_bert_input(batch[:3], device=self.device)

                doc_ids = batch[3].to(self.device)

                input_embs, output_embs, _, w_enc, attn_mask, loss_vq, perplexity = self.model(input_ids, attention_mask, valid_pos)  
                doc_enc, o_doc_enc,_ = self.utils.calc_doc_enc(w_enc, input_ids = input_ids, doc_ids=doc_ids, attn_mask=attn_mask, max_len=max_len)  
                o_doc_enc = o_doc_enc.to(self.device)

                prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
                word_dists, word_dists_o, _, _ = self.model.forward_topic(doc_enc)

                kld_theta, recon_loss_e, recon_loss_o = self.model._loss_bow(doc_enc, o_doc_enc, word_dists, word_dists_o, prior_mean, prior_var,
                                                                             posterior_mean, posterior_var, posterior_log_var)    

                #loss = recon_loss_e + recon_loss_o + kld_theta 
                loss = recon_loss_e + kld_theta 

                loss.backward()               
                optimizer.step()
               
                total_loss_rec_e.update(recon_loss_e.item())
                total_loss_rec_o.update(recon_loss_o.item())
                total_loss_kld.update(kld_theta.item())

            
            print(f"Epoch {epoch}: loss recon_e = {total_loss_rec_e.avg:.4f} loss recon_o = {total_loss_rec_o.avg:.4f} loss kld = {total_loss_kld.avg:.4f}")
            
            npmi, diversity, l_mse,_, perplexity = self.inference(epoch, suffix="_"+str(epoch))
            tq = npmi*diversity # topic quality                
            print(f"Epoch {epoch}: npmi={npmi:.4f}: diversity={diversity:.4f}: tq={tq:.4f}")               
            
            with open(os.path.join(save_vq_path, str(epoch)+".pickle"), 'wb') as f:
                pickle.dump((npmi, diversity), f)
            if tq > tq_best and not math.isnan(l_mse):                    
                print(f"Pretrained model saved to {save_vq_path}")
                torch.save(self.model.state_dict(), os.path.join(save_vq_path, "best.pt"))
                tq_best = tq
                npmi_best = npmi
                diversity_best = diversity                    
                best_epoch = epoch

            with open(os.path.join(save_vq_path, "best.pickle"), 'wb') as f:
                pickle.dump((nmi_best, purity_best, npmi_best, diversity_best, self.tw, self.td, best_epoch), f)          
           
            print(f"Current best npmi = {npmi_best:.4f}")
        print(f"Saved: npmi = {npmi_best:.4f} diversity = {diversity_best:.4f} tq= {tq_best:.4f} at epoch {best_epoch}")
      
    # initialize topic embeddings via LDA, or other BoW based models
    def inference(self, epoch, suffix="", test_mode=False):

        loss_mse_ret, loss_vq_ret, perplexity_ret = -1.0, -1.0, -1.0

        _, pretrained_vq_path = self.load_pretrained_path()
        if test_mode and not os.path.exists(pretrained_vq_path) and epoch>=0:
            print(f"pt files in {pretrained_vq_path} required")
            return 
        elif test_mode and epoch>=0:
            pth_path = os.path.join(pretrained_vq_path, str(epoch)+".pt")
            print(f"Loading pretrained model from {pth_path}")
            self.model.ae.load_state_dict(torch.load(pth_path)) 
        else:
            print('inference the model inplace')       
        
        #lda fitting for documents
        topics = None
        
        #get embedding for topic update                    
        result = self.model.get_info(vocab=self.utils.inv_vocab_gensim)
        tw = result['topic-word-matrix']
        td = result['topic-document-matrix'].transpose()
        topics = result['topics']

        self.tw = tw
        self.td = td
                       
        #Calculating measures      
        (measure, diversity), _ = self.utils.topic_inference_etm(topics, self.res_dir, suffix=suffix, measure="c_npmi")
        
        
        return measure, diversity, loss_mse_ret, loss_vq_ret, perplexity_ret   

    def inference_cluster(self, epoch=-1, test_mode=False):   
        
        self.model.to(self.device)

        #setting pretrained vq path
        _, pretrained_vq_path = self.load_pretrained_path()

        if test_mode and not os.path.exists(pretrained_vq_path) and epoch>=0:
            print(f"pt files in {pretrained_vq_path} required")
            return 
        elif test_mode and epoch>=0:
            pth_path = os.path.join(pretrained_vq_path, str(epoch)+".pt")
            print(f"Loading pretrained model from {pth_path}")
            self.model.load_state_dict(torch.load(pth_path)) 
        elif test_mode and epoch<0:
            pth_path = os.path.join(pretrained_vq_path, "best.pt")
            print(f"Loading pretrained model from {pth_path}")
            self.model.load_state_dict(torch.load(pth_path), strict=False) 
            
        else:
            print('inference the model inplace') 

        sampler = SequentialSampler(self.data_test)
        dataset_loader = DataLoader(self.data_test, sampler=sampler, batch_size=self.batch_size)

        thetas = []
        self.model.eval()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataset_loader), total=len(dataset_loader), desc="testing cluster idx"):
                input_ids, attention_mask, valid_pos, max_len = self.utils.get_bert_input(batch[:3], device=self.device)

                doc_ids = batch[3].to(self.device)
                _, _, _, w_enc, attn_mask, _, _ = self.model(input_ids, attention_mask, valid_pos)  
                doc_enc, o_doc_enc,_ = self.utils.calc_doc_enc(w_enc, input_ids = input_ids, doc_ids=doc_ids, attn_mask=attn_mask, max_len=max_len)  
                 
                self.model.forward_topic(doc_enc, o_doc_enc)
                theta = self.model.topic_model_bow.get_theta(doc_enc)
                thetas.append(theta)
        
        tds = torch.cat(thetas, dim=0)

        result = self.model.get_info(vocab=self.utils.inv_vocab_gensim)
        tw = result['topic-word-matrix']
        td = result['topic-document-matrix'].transpose()
        topics = result['topics']

        self.tw = tw
        self.td = td
                       
        #Calculating measures     
        (measure, diversity), _ = self.utils.topic_inference_etm(topics, self.res_dir, suffix="", measure="c_npmi")     
        nmi, purity = self.utils.km_cluster_eval(label_path=self.label_dir, preds=tds, n_clusters=self.args.n_clusters, split=[self.split[0],self.split[1]])       

        print(f"nmi={nmi:.4f} purity={purity:.4f} npmi={measure:.4f} diversity={diversity:.4f}")    
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='20ng', help='nyt, 20ng, m10, dblp, bbc_news, imdb')
    parser.add_argument('--dataset_base', default='datasets')
    parser.add_argument('--result_base', default='results')
    parser.add_argument('--model_selection', default='tvq_vae')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clusters', default=50, type=int, help='number of topics')
    parser.add_argument('--n_embeddings', default=300, type=int, help='number of expanded words')
    parser.add_argument('--alpha_hidden', default=1, type=int, help='alpha hidden')
    parser.add_argument('--k', default=25, type=int, help='number of top words to display per topic')
    parser.add_argument('--n', default=5, type=int, help='number of expanded words for each word')
    parser.add_argument('--input_dim', default=768, type=int, help='embedding dimention of pretrained language model')
    parser.add_argument('--kappa', default=10, type=float, help='concentration parameter kappa')
    parser.add_argument('--hidden_dims', default='[500, 500, 1000, 100]', type=str)

    parser.add_argument('--do_pretrain', action='store_true')
    parser.add_argument('--do_cluster', action='store_true')
    parser.add_argument('--do_inference', action='store_true')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs for clustering')
    parser.add_argument('--label_path', default='labels.txt')

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = TVQVAETrainer(args)      
 
    if args.do_pretrain:
        trainer.pretrain(train_epoch=args.epochs)
    else:
        if args.do_cluster:
            trainer.train(train_epoch=args.epochs)
        if args.do_inference:
            model_path = os.path.join("datasets", args.dataset, "model.pt")
            try:
                trainer.model.load_state_dict(torch.load(model_path))
            except:
                print("No model found! Run clustering first!")
                exit(-1)
            trainer.inference(topk=args.k, suffix=f"_final")
            trainer.inference_cluster(test_mode=True)
    
