# Medical Cross-Modality Domain Adaptation (Med-CMDA)

Here are implementations for paper: <br />

**PnP-AdaNet: Plug-and-Play Adversarial Domain Adaptation Network with a Benchmark at Cross-modality Cardiac Segmentation.** (https://arxiv.org/abs/1812.07907) (long version)
 
**Unsupervised Cross-Modality Domain Adaptation of ConvNets for Biomedical Image Segmentations with Adversarial Loss, IJCAI, pp. 691-697, 2018.** (https://arxiv.org/abs/1804.10916) (short version)

![](assets/README-83d896ef.png)

### Introduction

Deep convolutional networks have demonstrated the state-of-the-art performance on various medical image computing tasks. Leveraging images from different modalities for the same analysis task holds clinical benefits. However, the generalization capability of deep models on test data with different distributions remain as a major challenge. In this paper, we propose the PnPAdaNet (plug-and-play adversarial domain adaptation network) for adapting segmentation networks between different modalities of medical images, e.g., MRI and CT. We propose to tackle the significant domain shift by aligning the feature spaces of source and target domains in an unsupervised manner. Specifically, a domain adaptation module flexibly replaces the early encoder layers of the source network, and the higher layers are shared between domains. With adversarial learning, we build two discriminators whose inputs are respectively multi-level features and predicted segmentation masks. We have validated our domain adaptation method on cardiac structure segmentation in unpaired MRI and CT. The experimental results with comprehensive ablation studies demonstrate the excellent efficacy of our proposed PnP-AdaNet. Moreover, we introduce a novel benchmark on the cardiac dataset for the task of unsupervised cross-modality domain adaptation. We will make our code and database publicly available, aiming to promote future studies on this challenging yet important research topic in medical imaging.

### Usage

#### 0. Requirements

```
nibabel==2.1.0
nilearn==0.3.1
numpy==1.13.3
tensorflow-gpu==1.4.0
```

#### 1. Data preprocessing

The data repository is under construction.

For the ease of training, after data augmentation, training samples are expected to be written into `tfrecord` with the following format:
```python
feature = {
            # image size, dimensions of 3 consecutive slices
            'dsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
            'dsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
            'dsize_dim2': tf.FixedLenFeature([], tf.int64), # 3
            # label size, dimension of the middle slice
            'lsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
            'lsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
            'lsize_dim2': tf.FixedLenFeature([], tf.int64), # 1
            # image slices of size [256, 256, 3]
            'data_vol': tf.FixedLenFeature([], tf.string),
            # label slice of size [256, 256, 1]
            'label_vol': tf.FixedLenFeature([], tf.string)}
```

#### 2. Training base segmentation network

Run `train_segmenter.py`, where training configurations are specified.  

This calls `source_segmenter.py`, where network structure and training function are defined.

#### 3. Training adversarial domain adaptation

##### 3.1 Warming-up the discriminator

To obtain a good initial estimation of Wasserstein distances between feature maps of two domains, we first warm-up the feature domain discriminator. In order to do this, run

`python train_gan.py --phase warming-up`

#### 3.2 Training adversarial domain adaptation

After warming the discriminator up, we can then jointly train the feature domain discriminator and the domain adaptation module (generator). To do this, run

`python train_gan.py --phase train-gan`

The experiment configurations can be found in `train_gan.py`.  It calls `adversarial.py`, where network structures and training functions are defined.

#### 4. Evaluation

\# TODO: combine the testing code with training code and switch them with an additional argument

**Note: Repository still under construction ...**

Any questions on implementations, please email qi.dou@imperial.ac.uk (Qi Dou) and c.ouyang@imperial.ac.uk (Cheng Ouyang) <br />
Any questions on data or other general issues, please contact qi.dou@imperial.ac.uk (Qi Dou) and zxh@fudan.edu.cn (Xiahai Zhuang)
