# CV_Interview_Answers
This repository contains the implementations and answers to popular Computer Vision questions

An excellent [resource](https://arxiv.org/ftp/arxiv/papers/2201/2201.00650.pdf) for QnA rounds. You may also refer to [this](https://ml-notes-rajatgupta.notion.site/ml-notes-rajatgupta/47bf08f60cad49ba83c0675b0a360f6a?v=b9ba5ea7dbf64a2c84f4e8ebac4ba70bhttps://ml-notes-rajatgupta.notion.site/ml-notes-rajatgupta/47bf08f60cad49ba83c0675b0a360f6a?v=b9ba5ea7dbf64a2c84f4e8ebac4ba70b)

### Some Terminology

1. What is feature space?

2. What is Latent space?

3. What is embedding space?

4. What is representation space?

5. What are latent features?

6. What is a feature embedding?

7. What is feature representation?

7. What does latent representation mean?

8. What does embedding representation mean?

9. What does latent embedding refer to?

10. What does vector refer to?

11. What does Domain Distribution mean?

### Maths

1. What is [covariance](https://towardsdatascience.com/covariance-and-correlation-321fdacab168)?

2. What is correlation?

3. Explain the differences betn 1 and 2.

4. What do norms refer to?

5. Differences between 'distances' and norms?

6. [PCA](https://www.linkedin.com/feed/update/urn:li:activity:6862677707273199616?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A6862677707273199616%29)

7. SVD

8. K-Means - [overview](https://medium.com/@rishit.dagli/build-k-means-from-scratch-in-python-e46bf68aa875) and [mathametical](https://medium.com/analytics-vidhya/k-means-clustering-optimizing-cost-function-mathematically-1ccae156299f) explanation. 

8. L1 vs L2?

TLDR; Use L1 when there are no extreme outliers in the data otherwise in all other cases use L2. 

### Training and Accuracy Metrics

1. What is Precision?

Ans: Precision is 

2. What is Recall?

3. What is F1 score?

4. Define Confusion Matrix.

5. Define Bias and Variance.

6. How does model performace vary with bias and variance?

7. What does the ROC curve represent?
Ans: [here](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20(receiver%20operating,False%20Positive%20Rate))

8. How does the bias and variance vary with precision and recall?

9. What is the difference between test and validation sets? [Prelim idea here](https://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set)

10. Are validation sets always needed?

11. What is K-cross fold validation? [Ans](https://www.analyticsvidhya.com/blog/2022/02/k-fold-cross-validation-technique-and-its-essentials/)

### Layer Functions

1. What is Normalization?

2. What is Batch Normalization?

2. What is Instance Normalization?

### Activations and Losses 

[Nice Blog](https://rohanvarma.me/Loss-Functions/)

1. List down all standard losses and activations.

* Sigmoid
* ReLU
* Leaky ReLU
* Tanh
* Hard Tanh
* Cross Entropy 
* Binary Cross Entropy
* Kullback leibler divergence loss
* Triplet  - Bring centroid closer to mean (anchor)
* Hard Triplet Mining - Bring extreme points closer to mean (point)

![image](https://user-images.githubusercontent.com/55347156/143975872-0bcf2f0e-3f94-44ba-9a83-0490a9022fa9.png)


### Model Compression

1. What is Knowledge Distillation?

2. What is model pruning?

3. This awesome [twitter thread](https://twitter.com/MishaLaskin/status/1546641202900082688?s=20&t=_0EDwXH88ssxNYxGoTQwvQ) on model memory consumption.

4. How tensors are [stored in memory](https://twitter.com/francoisfleuret/status/1575756258669662208?s=20&t=V7Vfka9JIIFxJuMZhVF5DA)

### Some Standard Architectures
[Read here.](https://theaisummer.com/cnn-architectures/?utm_content=204647940&utm_medium=social&utm_source=linkedin&hss_channel=lcp-42461735)

1. VGG-16/19/152

2. Resnet - 18/50/150

* Skip connection:
* Identity connections:

3. Inception - v1/v2/v3

* Group convolution:

4. [Xception](https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568)
* Depthwise separable conv:
* Pointwise separable conv:

5. [MobileNet](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69)


### Convolution

1. What is convolution?

2. What are kernels/filters?

3. What is stride and padding?

4. [Derive the factor of improvement of depthwise separable conv over normal convolution.](https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568)


### Model Deployment and production

[Beginner's Blog](https://stackoverflow.blog/2020/10/12/how-to-put-machine-learning-models-into-production/) 

1. Data Drift vs Model Drift vs Concept Drift?

Extensive [repo](https://github.com/ahkarami/Deep-Learning-in-Production) on this topic

### Transformers and Attention

1. [Thoughts on Transformers](https://twitter.com/karpathy/status/1582807367988654081?s=20&t=pK-Uu90gjXRJ6fARyIVDKg) by Karpathy.

### Other cool stuff

1. [Hands on Stable Diffusion](https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing)

### Must Read Texts
1. Image Processing, Analysis and Machine Vision - *Sonka, Boyle*
2. Deep Learning - *Bengio, Goodfellow* \
[Download links](https://drive.google.com/drive/folders/1yjtIYdt3fq_YYRrDZaf1P8y_P3CYt2Xi?usp=sharing)
