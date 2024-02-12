import torch
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import BertTokenizer
import os
import string
from nltk.tag import pos_tag
from sklearn.cluster import KMeans
import gensim
from gensim.models import CoherenceModel
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch.optim as optim

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EMADecayOptimizer(optim.Optimizer):
    def __init__(self, optimizer, decay=0.999):
        self.optimizer = optimizer
        self.decay = decay
        self.shadow_params = {}
        for group in optimizer.param_groups:
            for param in group['params']:
                self.shadow_params[id(param)] = param.clone()

    def step(self, closure=None):
        self.optimizer.step(closure)
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                self.shadow_params[id(param)].data = (self.decay * self.shadow_params[id(param)].data + (1.0 - self.decay) * param.data)  
                param.data = self.shadow_params[id(param)].data
       
class TVQVAEClusUtils(object):

    def __init__(self):
        pretrained_lm = 'bert-base-uncased'
        self.vocab = None
        self.inv_vocab = None
        self.tokenizer = None
        self.texts = None
        self.id2word = None
        self.corpus = None
        self.vocab_bert2gensim = None
        
    def get_bert_vocabs(self, vocab, inv_vocab, tokenizer):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.tokenizer = tokenizer

    def encode(self, docs, max_len=512, given_vocab=None):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks
    
    def filter_words(self, word_list):
        filtered_words = []
    
        for word in word_list:
            synsets = wordnet.synsets(word)
            pos_tags = [syn.pos() for syn in synsets]
        
            if 'v' in pos_tags or 'n' in pos_tags or 'a' in pos_tags:
                filtered_words.append(word)
    
        return filtered_words
    
    def lemmatize_words(self, word_list):
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = []
    
        for word in word_list:
            pos = 'v'  # 동사로 가정합니다 (기본값)
        
            # WordNet 동사 품사 태그 확인
            synsets = wordnet.synsets(word)
            pos_tags = [syn.pos() for syn in synsets]
        
            if 'v' in pos_tags:
                pos = 'v'
            elif 'n' in pos_tags:
                pos = 'n'
            elif 'a' in pos_tags:
                pos = 'a'
            elif 'r' in pos_tags:
                pos = 'r'

            lemmatized_word = lemmatizer.lemmatize(word, pos)
        
            lemmatized_words.append(lemmatized_word)
    
        return lemmatized_words

    def create_dataset(self, dataset_dir, text_file, loader_name, vocab_name="vocab.pickle", doc_name= "corpus.pickle", max_len=512):
        loader_file = os.path.join(dataset_dir, loader_name)   
        vocab_file = os.path.join(dataset_dir, vocab_name)
        doc_file = os.path.join(dataset_dir, doc_name)
        stop_words = set(stopwords.words('english'))

        if os.path.exists(loader_file):
            if os.path.exists(doc_file):                 
                with open(doc_file, 'rb') as f:
                    docs = pickle.load(f)        
            else:                
                print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
                corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
                docs = []
                for doc in tqdm(corpus.readlines()):
                    content = nltk.word_tokenize(doc.strip())
                    content_tokenized = nltk.word_tokenize(doc.strip())
                    content_tokenized = [w.lower() for w in content_tokenized if not w.lower() in stop_words and w.isalpha() and w != 'br']
                    content_tokenized = self.lemmatize_words(self.filter_words(content_tokenized))
                    docs.append(content_tokenized)

                with open(doc_file, 'wb') as f:
                    pickle.dump(docs, f) 
            
            self.texts = docs 
            self.id2word = gensim.corpora.Dictionary(self.texts)        
            self.corpus = [self.id2word.doc2bow(text) for text in self.texts]     
            self.vocab_gensim = self.id2word.token2id     
            self.inv_vocab_gensim = {k:v for v, k in self.vocab_gensim.items()}  
            
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)    
            with open(vocab_file, 'rb') as f:
                vocab_bert2gensim = pickle.load(f)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = []
            docs_tokenized = []
            for doc in tqdm(corpus.readlines()):
                content = doc.strip()
                docs.append(content)

                content_tokenized = nltk.word_tokenize(doc.strip())
                content_tokenized = [w.lower() for w in content_tokenized if not w.lower() in stop_words and w.isalpha() and w != 'br']
                content_tokenized = self.lemmatize_words(self.filter_words(content_tokenized))
                docs_tokenized.append(content_tokenized)

            self.texts = docs_tokenized 
            self.id2word = gensim.corpora.Dictionary(self.texts)
            self.corpus = [self.id2word.doc2bow(text) for text in self.texts] 
            self.vocab_gensim = self.id2word.token2id   
            self.inv_vocab_gensim = {k:v for v, k in self.vocab_gensim.items()} 
            
            print(f"Converting texts into tensors.")
            input_ids, attention_masks = self.encode(docs, max_len)
            print(f"Saving encoded texts into {loader_file}")
            stop_words = set(stopwords.words('english'))
            filter_idx = []
            valid_pos = ["NOUN", "VERB", "ADJ"]
            
            vocab_bert2gensim = {}
            for i in self.inv_vocab:
                token = self.inv_vocab[i]
                
                if token in stop_words or token.startswith('##') \
                or token in string.punctuation or token.startswith('[') \
                or pos_tag([token], tagset='universal')[0][-1] not in valid_pos \
                or token not in self.vocab_gensim:
                    filter_idx.append(i)
                else:
                    j = self.vocab_gensim[token]
                    vocab_bert2gensim[i] = j
                    
            valid_pos = attention_masks.clone()
            for i in filter_idx:
                valid_pos[input_ids == i] = 0
            data = {"input_ids": input_ids, "attention_masks": attention_masks, "valid_pos": valid_pos}
            torch.save(data, loader_file) 
            with open(vocab_file, 'wb') as f:
                    pickle.dump(vocab_bert2gensim, f)
        
        self.vocab_bert2gensim = vocab_bert2gensim
        self.vocab_gensim2bert = {k:v for v, k in self.vocab_bert2gensim.items()} 

        return data
    
    def load_dataset(self, dataset_dir, loader_name, split=None):
        loader_file = os.path.join(dataset_dir, loader_name)
        assert os.path.exists(loader_file)
        print(f"Loading encoded texts from {loader_file}")
        data = torch.load(loader_file)

        if split is not None:
            data_train = {"input_ids": data['input_ids'][:split[0],:], 
                          "attention_masks": data['attention_masks'][:split[0],:], 
                          "valid_pos": data['valid_pos'][:split[0],:]}
            data_val = {"input_ids": data['input_ids'][split[0]:split[1],:], 
                        "attention_masks": data['attention_masks'][split[0]:split[1],:], 
                        "valid_pos": data['valid_pos'][split[0]:split[1],:]}
            data_test = {"input_ids": data['input_ids'][split[1]:,:], 
                        "attention_masks": data['attention_masks'][split[0]:split[1],:], 
                        "valid_pos": data['valid_pos'][split[1]:,:]}
        else:
            data_train = data
            data_val = None
            data_test = None                           

        return data_train, data_val, data_test
    
    def get_bert_input(self, batch, device):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        valid_pos = batch[2].to(device)
        max_len = attention_mask.sum(-1).max().item()
        
        input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in (input_ids, attention_mask, valid_pos))
        
        return input_ids, attention_mask, valid_pos, max_len
    
    def activation(self, matrix, mode='softmax'): 
        if mode =='softmax':
            exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
            softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
            return softmax_matrix
        elif mode == 'softplus':
            return np.log(1 + np.exp(matrix))
        else:
            return matrix   
        
    def normalize(self, x, axis=-1, eps=1e-12):
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + eps)

    def calc_doc_enc(self, w_encs, input_ids, doc_ids, attn_mask, max_len):
        n_words = max(self.vocab_bert2gensim.values())+1
        doc_id_expanded = doc_ids.unsqueeze(1).expand(doc_ids.shape[0], max_len)[attn_mask]

        o_doc_enc = []
        for input_id, attn in zip(input_ids, attn_mask):
            ids = input_id[attn]
            word_enc = torch.zeros(n_words)
            for id in ids:
                gensim_id = self.vocab_bert2gensim[id.item()]
                word_enc[gensim_id] += 1
            o_doc_enc.append(word_enc)
        o_doc_enc = torch.stack(o_doc_enc)

        doc_enc = []
        for doc_id in doc_ids:
            indices = (doc_id == doc_id_expanded)
            doc_enc.append(torch.sum(w_encs[indices], dim=0))
        doc_enc = torch.stack(doc_enc)

        return doc_enc, o_doc_enc, doc_id_expanded
    
    def purity(self, labels_true, labels_pred):
        clusters = np.unique(labels_pred)
        counts = []
        for c in clusters:
            indices = np.where(labels_pred == c)[0]
            max_votes = np.bincount(labels_true[indices]).max()
            counts.append(max_votes)
        return sum(counts) / labels_true.shape[0]
    
    def diversity(self, topics, topk=25, per_topic=True):
        if topics is None:
            return 0
        if topk > len(topics[0]):
            raise Exception('Words in topics are less than ' + str(topk))
        else:                                   
            if per_topic == False:
                unique_words = set()           
                for topic in topics:
                    unique_words = unique_words.union(set(topic[:topk]))
                return len(unique_words) / (topk*len(topics))
            else:
                diversities = []
                for topic in topics: 
                    unique_words = set()               
                    unique_words = unique_words.union(set(topic[:topk]))          
                    diversities.append(len(unique_words) / topk) 

                return diversities
            


    def measurement(self, topics_10, topics_25, measure='c_npmi', per_topic=True):          

        if measure in ['c_npmi', 'u_mass']:
            cm = CoherenceModel(topics=topics_10, dictionary = self.id2word, texts = self.texts, corpus=self.corpus, coherence=measure,topn=10)
            
            if per_topic == True:
                coherences = cm.get_coherence_per_topic()
                sorted_index = np.argsort(coherences)[::-1]
                measure_per_topic = sorted(coherences)[::-1]                
                len_measure = len(sorted_index)    
                topcis = [topics_25[t] for t in sorted_index[:len_measure//2]]
                diversity= self.diversity(topics=topcis, topk=25, per_topic=False)
                return sum(2*measure_per_topic[:len_measure//2]) / len_measure, diversity 
            else:
                diversity= self.diversity(topics=topics_25, topk=25, per_topic=False)
                return cm.get_coherence(), diversity                  
        else:
            return -1.0      

    def cluster_eval(self, label_path, preds, split=None, seed=42):
        labels = open(label_path).readlines()
        labels = np.array([int(label.strip()) for label in labels])

        if split is not None:
            labels= labels[split[0]:split[1]]

        _, indices = torch.topk(preds, k=1,dim=1)
        
        y_pred = indices.squeeze().cpu().numpy()
        
        nmi = normalized_mutual_info_score(labels, y_pred)
        purity = self.purity(labels, y_pred)
        
        return nmi, purity
    
    def km_cluster_eval(self, label_path, preds, n_clusters, split=None, seed=42):
        labels = open(label_path).readlines()
        labels = np.array([int(label.strip()) for label in labels])

        if split is not None:
            labels= labels[split[0]:split[1]]

        #run kmeans algorithms to get prediction
        pred_np = preds.cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pred_np)
        
        y_pred = kmeans.labels_
        
        nmi = normalized_mutual_info_score(labels, y_pred)
        purity = self.purity(labels, y_pred)
        
        return nmi, purity
    
        
    def topic_update(self, emb, w_enc, wds, wids, td, tw, n_clusters):
        
        #get word embeddings
        W = self.normalize(emb, axis=1)
        w_emb = self.normalize(np.matmul(w_enc, W))
        t_emb = self.normalize(np.matmul(tw, W))
        
        #update td_new
        wts_bert = self.activation(np.matmul(w_emb, t_emb.transpose()) - 1.0)
        
        td_new = np.zeros_like(td)
        tw_new = np.zeros_like(tw)
                    
        for w_e, wd, wid in zip(w_enc, wds, wids):
            wt = wts_bert[wid]

            #update td_new by accumulating wt for each w in a doc
            td_new[wd] += wt
            
            for t, val in enumerate(wt):
                tw_new[t] += val*w_e
        
        td_new /= td_new.sum(axis=1)[:, np.newaxis]
        tw_new /= tw_new.sum(axis=1)[:, np.newaxis]
        
        return td_new, tw_new
    
    def topic_inference(self, w_enc, wds, wids, td, tw, n_clusters, res_dir, suffix="", measure = "c_nmpi"):

        topic_file = open(os.path.join(res_dir, f"topics{suffix}.txt"), "w")
        wts = np.zeros((max(self.vocab_bert2gensim.values())+1, n_clusters))
        #tw = self.activation(tw)
        if td is not None:
            for w_e, wd, wid in zip(w_enc, wds, wids):
                tw_d = np.dot(np.diag(td[wd]),tw).transpose()
                #tw_d = tw.transpose()
                converted_wid = self.vocab_bert2gensim[wid]

                wt = np.dot(w_e, tw_d)
                wt = wt / np.sum(wt)
                wts[converted_wid,:] += wt
            wts = wts / (np.sum(wts, axis=1)[:,np.newaxis] + 1e-10)
            tws = wts.transpose()
            topic_freq = np.sum(tws, axis=1)
            freq_idx = np.argsort(-topic_freq)
            tws = tws[freq_idx]
            topics = np.argsort(-tws, axis=1)

            topics_10 = []
            topics_25 = []
            for t, topic in enumerate(topics):
                topic_ids_10 = topic[:10] 
                topic_ids_25 = topic[:25] 

                result_string_10 = []
                result_string_25 = []                

                for idx in topic_ids_10:
                    result_string_10.append(f"{self.inv_vocab_gensim[idx]}")
                topics_10.append(result_string_10)

                for idx in topic_ids_25:
                    result_string_25.append(f"{self.inv_vocab_gensim[idx]}")
                topics_25.append(result_string_25)                     
                                
                topic_file.write(f"Topic {t}: {','.join(result_string_25)}\n")
            
            return self.measurement(topics_10=topics_10, topics_25=topics_25, measure=measure), freq_idx   
            
        else:
            print('td not calculated')
            return -1
        
    def topic_inference_etm(self, topics_25, res_dir, suffix="", measure = "c_nmpi"):
            topic_file = open(os.path.join(res_dir, f"topics{suffix}.txt"), "w")
            topics_10 = []
            for t, topic_25 in enumerate(topics_25):
                topic_word_10 = topic_25[:10] 
                
                topics_10.append(topic_word_10)
                topic_file.write(f"Topic {t}: {','.join(topic_25)}\n")
            
            return self.measurement(topics_10=topics_10, topics_25=topics_25, measure=measure), 0

