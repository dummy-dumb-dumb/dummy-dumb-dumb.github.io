---
layout:       post
title:        "Very Deep Convolutional Networks for Large-Scale Image Recognition"
author:       
    - "박서연"
    - "김정현"
header-style: text
catalog:      true
tags:
    - CNN
    - VGG
    - ReLU
---

# 박서연

## Abstract

- Contribution
    - Investigate the effect of the conv network depth ****on its accuracy in the large-scale image recognition setting
        
        → Increase depth using very small(3x3) conv filters 
        
        → Push the depth to 16-19 weight layers 
        
    - Won 1st, 2nd place at ImageNet Challenge 2014

---

## 1. Introduction

- Important aspect of ConvNet architecture design → **depth**
- Increase the depth of the network by adding more conv layers → feasible due to the **use of very small(3x3) conv filters in all layers**
- Also applicable to other image recognition datasets

---

## 2. ConvNet Configurations

- Generic layout of ConvNet config & special config used in the eval

### Architecture
- Comparision w/ AlexNet
    ![AlexNet/VGGNet](/img/pics_11_22/AlexNetvs.png)
- VGG19 Config
    ![VGG](/img/pics_11_22/VGG.png)

|  | (output) size | filter size | # filter | padding  | stride | activation  |
| --- | --- | --- | --- | --- | --- | --- |
| input | 224 x 224 x 3 |  |  |  |  |  |
| conv3-64 | 224 x 224 x 64 | 3 x 3 | 64 | same | 1 | ReLU |
| conv3-64 | 224 x 224 x 64 | 3 x 3 | 64 | same | 1 | ReLU |
| maxpool | 112 x 112 x 64 | 2 x 2 |  |  | 2 |  |
| conv3-128 | 112 x 112 x 128 | 3 x 3 | 128 | same | 1 | ReLU |
| conv3-128 | 112 x 112 x 128 | 3 x 3 | 128 | same | 1 | ReLU |
| maxpool | 56 x 56 x 128 | 2 x 2 |  |  | 2 |  |
| conv3-256 | 56 x 56 x  256 | 3 x 3 | 256 | same | 1 | ReLU |
| conv3-256 | 56 x 56 x  256 | 3 x 3 | 256 | same | 1 | ReLU |
| conv3-256 | 56 x 56 x  256 | 3 x 3 | 256 | same | 1 | ReLU |
| conv3-256 | 56 x 56 x  256 | 3 x 3 | 256 | same | 1 | ReLU |
| maxpool | 28 x 28 x 256 | 2 x 2 |  |  | 2 |  |
| conv3-512 | 28 x 28 x 512 | 3 x 3 | 512 | same | 1 | ReLU |
| conv3-512 | 28 x 28 x 512 | 3 x 3 | 512 | same | 1 | ReLU |
| conv3-512 | 28 x 28 x 512 | 3 x 3 | 512 | same | 1 | ReLU |
| conv3-512 | 28 x 28 x 512 | 3 x 3 | 512 | same | 1 | ReLU |
| maxpool | 14 x 14 x 512 | 2 x 2 |  |  | 2 |  |
| conv3-512 | 14 x 14 x 512 | 3 x 3 | 512 | same | 1 | ReLU |
| conv3-512 | 14 x 14 x 512 | 3 x 3 | 512 | same | 1 | ReLU |
| conv3-512 | 14 x 14 x 512 | 3 x 3 | 512 | same | 1 | ReLU |
| conv3-512 | 14 x 14 x 512 | 3 x 3 | 512 | same | 1 | ReLU |
| maxpool | 7 x 7 x 512 | 2 x 2 |  |  | 2 |  |
| fc-4906 | 1 x 1 x 4906 |  |  |  |  | ReLU |
| fc-4906 | 1 x 1 x 4906 |  |  |  |  | ReLU |
| fc-1000 | 1 x 1 x 1000 |  |  |  |  | ReLU |
| softmax | 1 x 1 x 1000 |  |  |  |  |  |

- Input & pre-processing
    - Input: 224 x 224 RGB image
    - Pre-processing: Subtract mean RGB val from each pixel
- Conv & pooling layers
    - Pass img through a stack of conv layers which have **small receptive field 3 x 3** w/ stride = 1
    - Spatial pooling by 5 max-pooling layers which have 2 x 2 window w/ stride = 2
- fully-connected layers
    - 1st, 2nd fc layer: 4096 channels each
    - 3rd fc layer: 1000-way ILSVRC classification
    - Configs are all the same
- Final layer: soft-max layer
- Hidden layers: Equipped w/ the rectification non-linearity; ReLU
- Only one layer contain LRN (params are like LRN layer in AlexNet) → config A-LRN

### Configuration 
- A-E only differ in depth 
    ![ConfigAtoE](/img/pics_11_22/configAtoE.png)
    ![numParams](/img/pics_11_22/numParams.png)

### Discussion

- Difference w/ other nets
    - Very small 3 x 3 receptive fields
        - stack of three 3 x 3 conv layers has an effective receptive field of 7 x 7
            - Ex
                ![one 7 x 7](/img/pics_11_22/receptive7x7.png)
                ![three 3 x 3](/img/pics_11_22/receptive3x3.png)
                
        - Incorporate 3 non-linear rectification layers instead of a single one → make decision function more discriminative
        - Decrease # params
            - #Params needed for training three 3 x 3 filter size → 3^3 = 27
            - #Params needed for training one 7 x 7 filter size → 7^2 = 49
    - Incorporation of 1 x 1 conv layers (config C)
        - Can increase the non-linearity of the decision function w/o affecting the receptive fields of conv layers
        - It maintain spatial property & learn relations btw channels(RGB) → less computation
    - Comparison w/ GoogLeNet
        - GoogLeNet: top performing entry of the ILSVRC-2014 classification task
        - Both are based on very deep ConvNets (22 weight layers) & small conv filters (GoolLeNet used 3 x 3, 1 x 1, 5 x 5)
        - GoogLeNet’s network topology is more complex & spatial resolution of the feature maps is reduced more aggressively in the 1st layers to decrease the amount of computation
        - VGGNet is outperforming GoogLeNet in terms of single-net classification accuracy 

---

## 3. Classification Framework

- Details of classification ConvNet training & evaluation

### Training

- It generally follows AlexNet’s training procedure
- Used strategy
    - Optimize the multinomial logistic regression objective
    - Mini-batch gradient descent (batch size = 256)
    - Momentum = 0.9
    - Regularization by weight decay (L2 Norm)
    - Dropout regularization for the first two fc layers (dropout ratio: 0.5)
    - Initial learning rate = 10^(-2) → gradually decrease it by a factor of 10 when the val set accuracy stopped improving
        - Lr was decreased 3 times & learning stopped after 74 epochs
- Why does it need less epochs to converge than AlexNet?
    - Implicit regularization imposed by greater depth and smaller conv filter sizes
    - Pre-initialization of certain layers
- Initialization of weights
    1. Used random initialization on config A (which is shallow net)
    2. On deeper nets, initialize layers using net A’s layers (first 4 conv layers & last 3 fc layers) & use initial lr for pre-initialized layers but allow to change during learning 
    3. For random initialized layers, sample the weights from a normal distribution w/ 0 mean, 10^(-2) var 
    - Note: found later that it is possible to initialize the weights w/o pre-training → use random initialization procedure (Glorot & Bengio (2010))
        - Glorot initialization: initialize weight considering size of input and output
        - $W \sim \mathcal{N}(0, \frac{2}{n_{\text{in}} + n_{\text{out}}})$
- Obtaining fixed-size 224 x 224 ConvNet input image
    - Random crop from rescaled training images (1 crop per image per stochastic gradient descent iter)
    - Augmentations (both are used in AlexNet)
        - Random horizontal flipping
        - Random RGB color shift (PCA)
            - Reference from AlexNet
                - Form of DA #2: altering the intensities of the RGB channels in training images
                    - Perform PCA(Principal Component Analysis, 주성분 분석)on the set of RGB pixel vals throughout the ImageNet training set
                    - **To each training image, add multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues(고유값) times a random variable drawn from a Gaussian with mean 0 and standard deviation 0.1**
                    - (multiples of the found PC) ~ (corresponding eigenvalues) x (random variable)
                    - Eigenvalues?
                        - Calculated by covariance matrix(공분산 행렬) of RGB channel (공분산: 두 변수 간의 관계를 측정하는 통계적 지표)
                        - Represent “importance” of the principal directions of variation in the distribution of RGB values
                        
                        $$
                        \Sigma =
                        \begin{bmatrix}
                        \text{Var}(R) & \text{Cov}(R, G) & \text{Cov}(R, B) \\
                        \text{Cov}(G, R) & \text{Var}(G) & \text{Cov}(G, B) \\
                        \text{Cov}(B, R) & \text{Cov}(B, G) & \text{Var}(B)
                        \end{bmatrix}
                        $$
                        
                        - covariance matrix에 대해  고유분해(eigendecomposition)을 수행 → 행렬의 고유값(eigenvalues; 해당 방향의 중요도) & 고유벡터(eigenvectors; RGB 값 변화의 주요 방향) 얻을 수 있음
                    - How it works
                        - To each RGB image pixel ($I_{xy} = \begin{bmatrix} I_R \\ I_G \\ I_B \end{bmatrix}$),
                        - Add following quantity:  $\Delta I_{xy} = [p_1, p_2, p_3] \cdot [\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3]^T$
                            - $p_i, \lambda_i$: ith eigenvector and eigenvalue of the 3x3 covariance matrix of RGB pixel vals
                            - $\alpha_i$: random var drawn from a Gaussian w/ mean 0 and standard deviation 0.1 → only drawn again when the image is reused → w/ diff random var → gives variation to data
                    - This scheme captures an important property of natural images; object identity is invariant to changes in the intensity and color of the illumination
    - Training image rescaling
        - S?
            - S: val used to rescale the original image based on its shortest side (shortest dimension) before randomly cropping it to a fixed size of 224×224
            - isotropically rescaled
            - S should be ≥ 244
        - Setting training scale S  #1: fix S → single-scale training
            - S = 256, S = 384
            1. First train the net using S = 256 
            2. To speed up training of S = 384, initialize weights pre-trained w/ S = 256 & use smaller lr = 10^(-3)
        - Setting training scale S #2: multi-scale training
            - Each training img is individually rescaled by randomly sampling S from range [256, 512]
            - Objects in images can be a diff size → by training in various scales → better regularization performance
            - Can be seen as training set augmentation; **scale jittering**
            - **Train multi-scale models by fine-tuning all layers of a single-scale model with the same config, pre-trained w/ fixed S = 384**
            - Ex
                ![deer S = 256](/img/pics_11_22/deer256.png)
                ![deer S = 512](/img/pics_11_22/deer512.png)
                

### Testing

1. Isotropically rescale the input image to smallest image side; Q (test scale) 
    - S (training scale) → rescale by S, random crop 224 x 224
    - Q (test scale) → densely rescale by Q, using entire img (no need for multiple crops) → **Dense Evaluation**
2. FC layers converted to Conv layers → Fully-Convolutional Net 
    - 1st fc layer to 7 x 7 conv
    - 2nd, 3rd fc layers to 1 x 1 conv
    - Why?
        - FC layer depends on fixed input scale
        - FC layer flattens all input pixels → loose spatial info → can loose image context
        - By converting to conv layer → can deal with flexible input scale & utilize spatial info
        - Use trained weights of fc layers from trained ConvNet
    - Output: **class score map**
3. Class score map
    ![class score map](/img/pics_11_22/classScoreMap.png)

    - Class score map
        - Contain scores per class at each position of input img
        - Dimension: H x W x C
            - H, W resolution
            - C: # class = # ouput channel = 1000
    - Should obtain a fixed size vector
        - Spatially average (sum-pool) the class score map
        - H x W x C → 1 x 1 x C
    - Augment the test set by horizontal flipping (img1_original, img1_flipped) → get each softmax class posterior → average them → get final score of img1
4. Dense evaluation vs Multi-crop evaluation 
    - Multi-crop evaluation
        - Pros: can lead to improved accuracy
        - Cons: less efficient (too much computation)
    - Padding strategy
        - Dense eval: padding comes from the neighboring parts of an img → increase overall network receptive field → more context
        - Multi-crop eval: zero pad the boundary of each crop
    - Can be complementary
        
        $\text{Final Score} = \alpha \cdot \text{Dense Score} + \beta \cdot \text{Multi-Crop Score}$
        

### Implementation Details

- Framework: C++ Caffe toolbox + significant modifications
- Multiple GPUs (data parallelism)

---

## 4. Classification Experiments

- Dataset
    - Imgs of 1000 classes
    - Training(1.3M), validation(50K), testing(100K)
- Evaluation: top-1, top-5 error

### Single Scale Evaluation 
![single scale eval](/img/pics_11_22/singleTestScale.png)
- Test img size Q
    1. Fixed S → Q = S
    2. Jittered S → $Q = 0.5(S_{min} + S_{max})$
- A vs A-LRN
    - Using LRN does not show improvement
- Classification error decreases w/ the increase of ConvNet depth
    - B vs C
        - Diff # weight layers
        - Even tho C has 1 x 1 conv layers C performs better than B → additional non-linearity does help
    - C vs D
        - Have same # weight layers
        - C has three 1 x 1 conv layers, D has three 3 x 3 conv layers → D can capture more spatial context
    - Error rate saturate when the depth reaches 19 layers (config E)
    - Deeper depth can be helpful to bigger dataset
    - Net B w/ five 5 x 5 conv layers (shallow net)
        - Top-1 error of shallow net is 7% higher
        - Deep net w/ small filters outperforms a shallow net w/ larger filters
- Scale jittering at **training time** leads to significantly better results than fixed S

### Multi-Scale Evaluation 
![multi scale eval](/img/pics_11_22/multipleTestScale.png)
- Scale jittering
    1. Run a model over several rescaled vers of a test image (diff val of Q)
    2. Calculate class posterior of each rescaled img
    3. Average the resulting class posteriors
- W/ fixed S
    - Large discrepancy btw training/testing scales → drop in performance
    - Eval w/ 3 test img sizes → $Q = \{S − 32, S, S + 32\}$
- W/ jittered S
    - Can deal w/ wide range of Q
    - $Q = \{S_{min}, 0.5(S_{min} + S_{max}), S_{max}\}$
- Scale jittering at test time leads to better performance

### Multi Crop Evaluation
![multi crop eval](/img/pics_11_22/multipleCrop.png)
- Dense & multi-crop eval are complementary → combination outperforms each of them

### ConvNet Fusion
![convnet fusion](/img/pics_11_22/multipleFusion.png)
- Combine outputs of several models by averaging their softmax class posteriors 

### Comparison with the State of Art
![state of art](/img/pics_11_22/stateofArt.png)

---
## 5. Conclusion

- Evaluation of deep convolutional networks for large scale image classification

---
# 김정현
