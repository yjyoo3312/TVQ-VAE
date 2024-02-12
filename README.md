# TVQ-VAE
Topic-VQ-VAE: Leveraging Latent Codebooks for Flexible Topic-Guided Document Generation. AAAI 2024.

# Introduction
The code provides the implemenetation of the TVQ-VAE in three applications: document analysis, image generation using PixelCNN, and image generation with Transformer. You can check each implementation in the folders below.

```
-DOCUMENT_ANALYSIS
-IMAGE_GENERATION
-IMAGE_GENERATION_T
```

# Running the Code
In each folder, you can test the proposed TVQ-VAE. 

## Document Analsis
### Dataset
The implementation solves BoW style topic extraction from documents. We provide preprocessed data for 20NG and NYT datasets.
Download the preprocessed.zip file and extract it in each folder.

[20ng](https://drive.google.com/file/d/1MtJU_LVB2Sn19G8P3eGYlJltpNwsi4Qb/view?usp=sharing), [NYT](https://drive.google.com/file/d/1uBkiGarwKNmjAX-81ozL3hKeqocmQSCa/view?usp=sharing)


```
-DOCUMENT_ANALYSIS
 -datasets
   -20ng
   -nyt
   -yourown
```

You can also test your dataset. If you have the `corpus.txt` file in `yourown` folder, our code will do preprocessing in its first pretrain run.

### Training

The training code consists of two phases: pretraining of the VQ-VAE, and the training of the TVQ-VAE.
We provide the pretrained VQ-VAE code for 20NG and NYT datasets in the below link.

[20ng](https://drive.google.com/drive/folders/1C-1pz5muuvoPoLWvn7poimVXI5Xrg1m2?usp=sharing), [NYT](https://drive.google.com/drive/folders/1fOiLe8wdteDaF0ZB0MuGLqNMnakdU7oB?usp=sharing)

In the link, you can access the folder `tvq_vae`, consisting of 

```
-DOCUMENT_ANALYSIS
 -tvq_vae
   -alpha_hidden_1
   -alpha_hidden_1
   -pretrained_vq_300_5
```

Among the folders, `pretrained_vq_300_5` contains the pretrained vq_vae weights, which having embedding of 300 and n of 5. Place the `tvq_vae` folder under the dataset folder, as
```
-DOCUMENT_ANALYSIS
 -datasets
   -20ng
    -tvq_vae
     -...
     -pretrained_vq_300_5
       -best.pt
   -nyt
   -yourown
```
Then, we can train our TVQ-VAE model, from

```
python3 trainer.py --dataset nyt --n_clusters 10 --n_embeddings 300 --epochs 0 --lr 1e-3  --seed 1 --do_cluster --n 5 --alpha_hidden 1 --model_selection tvq_vae
```
Parameter `alpha_hidden` denotes the various settings for topic-word generation module. `alpha_hidden=1` provides the equivalent setting to the paper.
You can use `train.sh` file, as well. We note that `epochs` is set to zero by default, because the epoch denotes the training iteration to modify the VQ-VAE part, which is basically not an option in our method.

For pretraining, you can use
```
python3 trainer.py --do_pretrain --dataset '20ng' --n_embeddings 300 --n 5 --epochs 1000
```
We note that we set all the example setting to embeeding of 300 and n of 5. You can use `prtrain.sh` file for running it.

### Evaluation
This implementation includes improved topic extractions, even compared to the initial version of the paper. The modifications on optimizer, epochs, and learning rates significantly improved the topic quality. Check our impolementation for more detail.
You can get the quantitative results as:

#### 20NG
Initial version.

| Topic | NPMI 50%          | Diversity          | TQ           |
|-------|-------------------|--------------------|--------------|
| 10    | 0.1729            | 0.9920             | 0.1715       |
| 20    | 0.1710            | 0.9360             | 0.1601       |
| 30    | 0.1752            | 0.8907             | 0.1561       |
| 40    | 0.1933            | 0.7980             | 0.1543       |
| 50    | 0.1855            | 0.7600             | 0.1410       |
| Avg   | 0.1796            | 0.8753             | 0.1566       |

**alpha_hidden=1**

| Topic | NPMI 50%          | Diversity          | TQ           |
|-------|-------------------|--------------------|--------------|
| 10    | 0.1803            | 0.9920             | 0.1788       |
| 20    | 0.1973            | 0.8960             | 0.1768       |
| 30    | 0.2004            | 0.8987             | 0.1801       |
| 40    | 0.1984            | 0.8220             | 0.1631       |
| 50    | 0.1625            | 0.7328             | 0.1191       |
| Avg   | 0.1878            | 0.8683             | 0.1636       |

alpha_hidden=2

| Topic | NPMI 50%          | Diversity          | TQ           |
|-------|-------------------|--------------------|--------------|
| 10    | 0.1799            | 0.9920             | 0.1784       |
| 20    | 0.1976            | 0.9160             | 0.1810       |
| 30    | 0.2126            | 0.8853             | 0.1882       |
| 40    | 0.2029            | 0.7960             | 0.1615       |
| 50    | 0.1530            | 0.7184             | 0.1099       |
| Avg   | 0.1892            | 0.8615             | 0.1638       |

#### NYT
Initial version.

| Topic | NPMI 50%          | Diversity          | TQ           |
|-------|-------------------|--------------------|--------------|
| 10    | 0.1297            | 0.9840             | 0.1715       |
| 20    | 0.1585            | 0.9320             | 0.1601       |
| 30    | 0.1640            | 0.9840             | 0.1561       |
| 40    | 0.1564            | 0.9380             | 0.1467       |
| 50    | 0.1395            | 0.9700             | 0.1353       |
| Avg   | 0.1496            | 0.9616             | 0.1437       |

**alpha_hidden=1**

| Topic | NPMI 50%          | Diversity          | TQ           |
|-------|-------------------|--------------------|--------------|
| 10    | 0.1590            | 1.0000             | 0.1590       |
| 20    | 0.1876            | 0.9960             | 0.1868       |
| 30    | 0.2013            | 0.9947             | 0.2002       |
| 40    | 0.2048            | 0.9760             | 0.1999       |
| 50    | 0.1885            | 0.9392             | 0.1770       |
| Avg   | 0.1882            | 0.9881             | 0.1846       |

alpha_hidden=2

| Topic | NPMI 50%          | Diversity          | TQ           |
|-------|-------------------|--------------------|--------------|
| 10    | 0.1420            | 1.0000             | 0.1420       |
| 20    | 0.1963            | 1.0000             | 0.1963       |
| 30    | 0.2104            | 0.9947             | 0.2093       |
| 40    | 0.2021            | 0.9780             | 0.1977       |
| 50    | 0.1894            | 0.9472             | 0.1794       |
| Avg   | 0.1880            | 0.9840             | 0.1849       |

### References

We start implementing our code based on [Topclus](https://github.com/yumeng5/TopClus). We appreciate the sharing.

## Image Generation - PixelCNN

This example tests the proposed TVQ-VAE for image generation using CIFAR10 and CelebA dataset.

### Dataset

We provide two datasets: CIFAR10 and CelebA datasets. Please refer the link [1](https://www.cs.toronto.edu/~kriz/cifar.html), [2](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to get each ciations.

```
-IMAGE_GENERATION
 -data
   -celeba
   -cifar10
```

Then, unzip each dataset into the `data` folder.

### Training

To train the model, we provide VQ-VAE pretrained models for CIAR10 and CelebA datasets.

[CIFAR10](https://drive.google.com/drive/folders/199GtQuacvA1JBRUTKrEHZ1KMu5gCuRGq?usp=sharing), [CelebA](https://drive.google.com/drive/folders/1_k9yuio3wFbHAvRPlYoAAaw_EuLPmHPx?usp=sharing)

Then, place the pretrained VQ-VAE models in

```
-IMAGE_GENERATION
 -data
 -models
  -vqvae_celeba
   -best.pt
  -vqvae_cifar10
   -best.pt
```
Now, you can train the tvq_vae by following the command:

```
python3 train_tvqvae_e2e.py --dataset 'cifar10' --n_clusters 100 --output-folder 'tvqvae_cifar10_e2e_100' --num-epochs 100 
```

`n_cluster` denotes the number of topics. You can use `run_tvqvae_e2e_celeba.sh` and `run_tvqvae_e2e_cifar10.sh`.

### Generation

We provide two generation script: one for visualizing the topics and the others for generating images based on the refernece image.
The former one can be done by `generatioin_pixelcnn_prior_topic_vis.py`, and the latters by `generation_pixelcnn_prior_i2i_e2e.py`.
See arguments for more detailed information. 

For i2i image generation, use the following command
```
python3 generation_pixelcnn_prior_i2i_e2e.py --dataset 'cifar10' --n_clusters 100 --vqvae_model 'models/tvqvae_cifar10_e2e_100/best_loss_prior.pt' --samples 'samples/cifar10_topic_e2e_i2i_100' 
```

For topic visualization, use the following command
```
python3 generation_pixelcnn_prior_topic_vis.py --dataset 'cifar10' --n_clusters 100 --vqvae_model 'models/tvqvae_cifar10_e2e_100/best_loss_prior.pt' --samples 'samples/cifar10_topic_prior'
```

You can use the pretrained TVQ-VAE weights in the links below:
[CelebA](https://drive.google.com/drive/folders/1ZOgq3ubIAvYcAwEfACjJpA4S4P7M74mg?usp=sharing), [CIFAR10](https://drive.google.com/drive/folders/1XtTRgVhjZsx3FPTUxnuK2vLimHOIHq2o?usp=sharing)

### Reference
We refer the pytorch implementation of [VQ-VAE](https://github.com/ritheshkumar95/pytorch-vqvae) as our baseline code. We appreciate the sharing.

## Image Generation - Transformer

This example tests the proposed TVQ-VAE for image generation combined with [taming-transformer](https://github.com/CompVis/taming-transformers).

### Dataset

We follow the equivalent dataset settings to those in taming transformer repo. Our code tests facesHQ. 
Hence, following the [repository](https://github.com/CompVis/taming-transformers), download CelebHQ and FFHQ datasets and configure the datasets as instructed.

### Training

You can test our code with 

```
python3 main.py --base configs/faceshq_transformer_tvq.yaml -t True --gpus 1 --max_epochs 12
```

with the pretrained `2020-11-09T13-33-36_faceshq_vqgan` weight files. You can download the files from [here](https://drive.google.com/drive/folders/1-YBO-OdFQ1CHvD7s2Kw2jOJzCCIt2u6G?usp=sharing), and place it into the `'logs` folder. You can check other hyperparameters in `configs/faceshq_transformer_tvq.yaml`. 

### Sampling

You can sample the images from 

```
python3 make_samples_tvq.py --base configs/faceshq_transformer_tvq.yaml --resume logs/2024-02-11T14-54-05_faceshq_transformer_tvq/ --temperature 0.99 --outdir results --sample_size 16 --reference_size 16
```

We provide the pretrained tvq-vae files and example samples in [pretrained](https://drive.google.com/drive/folders/1iYucHlFBnq_kL0vwmpoHi9OPCgB1G-HQ?usp=sharing), [samples](https://drive.google.com/drive/folders/1wxckzrdVfdZAo6Nk7a6qoLv4uAqhq5V8?usp=sharing). 

### References

We implemented our TVQ-VAE code integrated with Transformer from [taming-transformer](https://github.com/CompVis/taming-transformers). 


# Citation

@article{yoo2023topic,
  title={Topic-VQ-VAE: Leveraging Latent Codebooks for Flexible Topic-Guided Document Generation},
  author={Yoo, YoungJoon and Choi, Jongwon},
  journal={arXiv preprint arXiv:2312.11532},
  year={2023}
}