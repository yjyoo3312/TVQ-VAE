# TVQ-VAE
Topic-VQ-VAE: Leveraging Latent Codebooks for Flexible Topic-Guided Document Generation. AAAI 2024.

# Introduction
he code includes the implementation of TVQ-VAE across three applications: document analysis, image generation using PixelCNN, and image generation with Transformer. You can explore each implementation in the respective folders provided below.

```
-DOCUMENT_ANALYSIS
-IMAGE_GENERATION
-IMAGE_GENERATION_T
```

# Running the Code
You can test the proposed TVQ-VAE within each respective folder.

## Document Analsis
### Dataset
The implementation addresses BoW-style topic extraction from documents. We offer preprocessed data for the 20NG and NYT datasets. Please download the `preprocessed.zip` file and extract it into each corresponding folder.

[20ng](https://drive.google.com/file/d/1MtJU_LVB2Sn19G8P3eGYlJltpNwsi4Qb/view?usp=sharing), [NYT](https://drive.google.com/file/d/1uBkiGarwKNmjAX-81ozL3hKeqocmQSCa/view?usp=sharing)


```
-DOCUMENT_ANALYSIS
 -datasets
   -20ng
   -nyt
   -yourown
```

If you possess the `corpus.txt` file within the `yourown` folder, our code will conduct preprocessing during its initial pretraining run, allowing you to test your dataset as well.

### Training

The training code comprises two phases: pretraining the VQ-VAE and training the TVQ-VAE. We offer the pretrained VQ-VAE code for the 20NG and NYT datasets at the link below.

[20ng](https://drive.google.com/drive/folders/1C-1pz5muuvoPoLWvn7poimVXI5Xrg1m2?usp=sharing), [NYT](https://drive.google.com/drive/folders/1fOiLe8wdteDaF0ZB0MuGLqNMnakdU7oB?usp=sharing)

Within the provided link, you'll find the folder labeled `tvq_vae`, which contains

```
-DOCUMENT_ANALYSIS
 -tvq_vae
   -alpha_hidden_1
   -alpha_hidden_1
   -pretrained_vq_300_5
```

Among the folders, `pretrained_vq_300_5` contains the pretrained VQ-VAE weights, with embeddings of size 300 and a neighborhood size of 5. Please place the `tvq_vae` folder within the dataset directory, as

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
Following that, we can proceed to train our TVQ-VAE model, starting from

```
python3 trainer.py --dataset nyt --n_clusters 10 --n_embeddings 300 --epochs 0 --lr 1e-3  --seed 1 --do_cluster --n 5 --alpha_hidden 1 --model_selection tvq_vae
```
The parameter `alpha_hidden` represents different configurations for the topic-word generation module. Setting `alpha_hidden=1` corresponds to the configuration described in the paper. You can also utilize the `train.sh` file. It's worth mentioning that `epochs` is typically set to zero by default, as epochs denote training iterations for adjusting the VQ-VAE component, which isn't applicable in our approach.

For the pretraining phase, you have the option to utilize
```
python3 trainer.py --do_pretrain --dataset '20ng' --n_embeddings 300 --n 5 --epochs 1000
```
We note that we've configured all examples to use embeddings of size 300 and a neighborhood size of 5. To execute the pretraining, you can utilize the `prtrain.sh` file.

### Evaluation
This implementation features enhanced topic extraction capabilities, surpassing even the initial version described in the paper. The adjustments made to the optimizer, epochs, and learning rates have notably enhanced the quality of topics generated. For further details, please refer to our implementation. Additionally, you can obtain quantitative results such as:

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
We have initiated the implementation of our code inspired by [Topclus](https://github.com/yumeng5/TopClus). We are grateful for the shared resources.

## Image Generation - PixelCNN

This example evaluates the proposed TVQ-VAE for image generation, utilizing the CIFAR10 and CelebA datasets.

### Dataset


We offer two datasets: CIFAR10 and CelebA datasets. You can refer to the following links to access each dataset: [1](https://www.cs.toronto.edu/~kriz/cifar.html), [2](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


```
-IMAGE_GENERATION
 -data
   -celeba
   -cifar10
```

Then, unzip each dataset into the `data` folder.

### Training

For training the model, we offer pretrained VQ-VAE models tailored for both the CIFAR10 and CelebA datasets.

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


We offer two generation scripts: one for visualizing topics and the other for generating images based on reference images.

To visualize topics, use `generation_pixelcnn_prior_topic_vis.py`. For image-to-image (i2i) generation, use `generation_pixelcnn_prior_i2i_e2e.py`. Please refer to the arguments for more detailed information.

For i2i image generation, execute the following command:

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
We acknowledge and appreciate the sharing of the PyTorch implementation of VQ-VAE available at this [link](https://github.com/ritheshkumar95/pytorch-vqvae), which serves as the baseline code for our work.

## Image Generation - Transformer

This example evaluates the proposed TVQ-VAE for image generation in conjunction with the [taming-transformer](https://github.com/CompVis/taming-transformers).

### Dataset
We adhere to the dataset configurations equivalent to those specified in the taming transformer repository. Our code focuses on testing facesHQ. Therefore, following the instructions provided in the [repository](https://github.com/CompVis/taming-transformers), please download the CelebHQ and FFHQ datasets and configure the datasets accordingly.

### Training

You can test our code from 

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

We have integrated our TVQ-VAE code with the Transformer model from the [taming-transformer](https://github.com/CompVis/taming-transformers). We appreciate the resources shared.

# Citation

@article{yoo2023topic,
  title={Topic-VQ-VAE: Leveraging Latent Codebooks for Flexible Topic-Guided Document Generation},
  author={Yoo, YoungJoon and Choi, Jongwon},
  journal={arXiv preprint arXiv:2312.11532},
  year={2023}
}