# Implementation and Evaluation of Variational Autoencoders for High Resolution Medical Imaging
 
Over the last few years deep learning models have achieved great success to improve computer aided diagnosis. Most of the deep learning models are using supervised learning methods, which have the disadvantage of elaborate preprocessing. Therefore, this work is focusing on an unsupervised learning approach, namely, on variational autoencoders (VAE)s proposed by Kingma et al. [[1]](#1). VAEs are powerful generative models, which allow analyzing and disentanglement of the latent space for a given input. In this work the standard VAE model and four enhanced models are derived, applied and discussed on a high resolution knee dataset.

## Implemented and Evaluted Models
1. Variational Autoencoder (VAE) [[1]](#1)
2. Spatial Variational Autoencoder (SVAE) [[2]](#2)
3. Variational Perceptual Generative Autoencoders (VPGA) [[3]](#3)
4. Vector Quantized Variational Autoencoder (VQ-VAE) [[4]](#4)
5. Introspective Variational Autoencoder (IntroVAE) [[5]](#5)

## Structure of this repository
The directory general contains different util methods for the implementation. In the model folder the five different models are implemented using the same architecture. The figures and plots for the thesis are created with the utillity files in the thesis_util directory. 
## Playing around with the laten space of IntroVAE
prerequisites:  Python 3.7 installed
1. Install required python packages
```
$ pip install -r requirements.txt
```
2.  Start JupyterLab 
```
$ jupyter-lab
```
3. Navigate to Jupyter Notebook
```
models/IntroVae/IntroVAE_latent_space_interactive.ipynb
```
4. set the path of your project in the path2project variable
## References
<a id="1">[1]</a> 
Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. URL
https://arxiv.org/pdf/1312.6114.pdf.

<a id="2">[2]</a> 
ZhengyangWang, Hao Yuan, and Shuiwang Ji. Spatial variational auto-encoding via
matrix-variate normal distributions. URL http://arxiv.org/pdf/1705.06821v2.

<a id="3">[3]</a>
Zijun Zhang, Ruixiang Zhang, Zongpeng Li, Yoshua Bengio, and Liam Paull. Perceptual
generative autoencoders. URL http://arxiv.org/pdf/1906.10335v1.

<a id="4">[4]</a>
Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation
learning, . URL http://arxiv.org/pdf/1711.00937v2.

<a id="5">[5]</a>
Huaibo Huang, Zhihang Li, Ran He, Zhenan Sun, and Tieniu Tan. Introvae:
Introspective variational autoencoders for photographic image synthesis. URL
http://arxiv.org/pdf/1807.06358v2.
