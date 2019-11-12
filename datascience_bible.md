# ML Algorithms
## Supervised-Classification
### Random Forest
#### Introduction
#### Pros and cons
#### Explanation
https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
#### Example
#### Case uses
#### Other

### Logistic Regression
### Naive Bayes
### Support Vector Machine
### kNeighbors
### AdaBoost
### Gradient Boost
Gradient boost is and ensemble classifier. This means that it works by iterativerily creating weak classifiers and using the ensemble of them to make a better prediction. Do not be confused with AdaBoost, even if both models have similiarities they are not quite the same.

In detail, Gradient Boost starts with a single predition that minimizes the totality of the loss functionj (e.g. the mean if using the RMSE) and then iterating through the folllowing process:
1. Calculate the gradient of the loss function (called residuals).
2. Create a "small" decision tree that predicts the negative of the residuals. Multiply the predictions of the tree by a scalar called learning rate.
3. Make new predictions by adding the contribution of the new decision tree to the previous predictions.
4. Stop once the maximum number of tress has been reached or if the new tree fails to improve performance enough.

The final predictions of the model will be the initial prediction summed to the scaled sum of all the trees.
####
### Hidden Markov Models
###

## Supervised-Regression
### 

## Clustering
### k-Means
#### Mini-batch kMeans
### DBSCAN
### OPTICS
### HAC family
#### Simple
#### Complete
#### UMPG
####
### Manifold
### MeanShift
###

### Fuzzy clustering
#### Fuzzy C-Means (FCM)
####
#### \try{FLAME}{https://en.wikipedia.org/wiki/FLAME_clustering}
###


## Dimensionality Reduction
### Factorial analyses
#### PCA
#### CA
#### MCA
#### FAMG

### Kernel PCA
### ICA
### LLE
### MDS
### TNSE
Stochastic Neighbor Embedding
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

## Recommendendation engine
### Collaborative filtering


# Metrics and Norms
## Error Metrics
### Accuracy
### Precision and recall
### F1 and F2 scores
### Beta score
### ROC
### RMSE
###
## Clustering metrics
### Silhouette
### Distance ratio
### Variance ratio
###
## Multi-category metrics
### Confusion matrix
### Entropy
###
## Norms
### Euclidan
### Manhattan
### Minkowski-p
### Mahalanobis 
### Hamming distance
### Damerau–Levenshtein
### Haversine
###
## Others
### Cramer's V
Cramer's V is measure of intercorrelation between 2 nominal (categorical) values. Its output value's range is [0, 1].
### Covariance
### Pearson Correlation
###
https://en.wikipedia.org/wiki/Uncertainty_coefficient


# DL Algorithms
## NN layer types
### Dense
### Convolutional
### Time delay NN
### Recurrent
#### LSTM
## Architecture types
### Classification
### Auto-Encoder
#### Tiramisu
#### Attention
#### Self attention

### Embeddings
### Generative Adversarial Network (GAN)
https://medium.com/sigmoid/a-brief-introduction-to-gans-and-how-to-code-them-2620ee465c30
https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b
###

## Famous Neural Networks
### Convolutional
#### VGG family (VGG16 and VGG19)
#### Inception family
#### ResNet
#### LeNet

### Recurrent
#### ELMo
#### BERT
https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3

# RL
https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419

# Data Preparation
https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779

# Statistics
http://www.randomservices.org/
## Distributions
### Bernoulli
### Binomial
### Hypergeometric
### Poisson
### Polya
### Exponential
### Gamma
### Normal
### Chi
### Uniform
### Beta
### Weibull
### NBD
### Power law
### ParetoNBD
https://mikulskibartosz.name/predicting-customer-churn-using-the-pareto-nbd-model-53f96fb142ef
https://mikulskibartosz.name/predicting-customer-lifetime-value-using-the-pareto-nbd-model-and-gamma-gamma-model-7e1b40a87888
###
## Methods
### Hypothesis tests
### Anova
### Bayesian Inference
#### MCMC
####
###

# Python Packages
## Numpy
## Scipy
## Pandas
## Scikit Learn
## Tensorflow
## PyTorch
## Keras
## lifetimes
## prince

# Spark
## Spark Introduction
## Spark SQL
### Spark SQL functions
#### `explode`
#### `when`
#### `array_\w*`
See Section Spark DataTypes, ArrayType.
####
### Built-ins
#### Columns
##### `cast`
####  Built-ins DF
##### `.withColumn`
##### `.limit`
##### `.toPandas`
##### `.crossJoin`
### Spark DataTypes
#### ArrayType
https://www.mungingdata.com/apache-spark/arraytype-columns
#### MapType
####