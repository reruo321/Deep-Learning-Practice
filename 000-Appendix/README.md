# Statistical Inference
## Naive Bayes
**Naive Bayes classifier** is one of simple probabilistic classifiers based on applying Bayes' theorem with *strong (naive) independence assumptions between the features*. Bias ↑, Variance ↓.

(Easy Guide: https://www.youtube.com/watch?v=O2L2Uv9pdDA)

### Bayes' Theorem
**Bayes' theorem** describes the probability of an event, based on prior knowledge of conditions that might be related to the event.

![000bayer](https://user-images.githubusercontent.com/48712088/203980793-6985a66f-bd60-40fc-8486-5160126a23ab.jpg)

#### Prior Probability
**Prior probability** is the probability distribution that would express one's belief, in Bayesian statistical inference.
#### Likelihood
**Likelihood** is the probability of something discrete and individual. (excluding something continuous like weight or height)

# Bias & Variance

## 1. Bias
**Bias** is the inability for a machine learning method to capture the true relationship.

## 2. Variance
**Variance** is the amount by which the prediction would change, if we fit the model to a different training data set.

### Consequence of Variance
**Consequence of Variance** is the difference in fits between data sets.

## A. Overfitting
**Overfitting** is the production of an analysis too fitting to a particular set of data → may fail to fit to additional data or predict future observations reliably. When a complicatied model too fits to a training data, it is likely to have a higher rate of error on new unseen data.

![000overfitting](https://user-images.githubusercontent.com/48712088/203996143-2a91d684-46e4-486d-9bd0-da338a468a87.png)

### How to Prevent Overfitting?
1. Training Data ↑
2. Regularization
3. Drop-Out

## B. Underfitting
**Underfitting** occurs when a mathematical model cannot adequately capture the underlying structure of the data.

## Ensemble Learning
**Ensemble Learning** uses multiple learning algorithms → Obtain better predictive performance than just using one algorithm. Some examples are *Regularization*, *Boosting*, and *Bootstrap Aggregating*.

### 1. Regularization
**Regularization** is an ensemble learning method that changes the result answer to be "simpler": To obtain results for ill-posed problems OR to prevent overfitting.

How to: Make weights not too big (Too big weight = Too complicated, squiggly function), Change cost function (Smaller cost function = Specific weights ↑ = Bad result)
### 2. Boosting
**Boosting** is one of the ensemble methods that combines a set of sequential weak learners into a strong learner to minimize training errors.

Step: 1st learner classifies data → Weight is given to misclassified data → 2nd learner classifies data → Weight is given to misclassified data → 3rd learner classifies data → Combine all classifiers → Final prediction

(Guide: https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
### 3. Bootstrap Aggregating
**Bootstrap Aggregating**, or **Bagging** is an ensemble meta-algorithm(= metaheuristic, a higher-level heuristic designed to find, generate, or select a heurisitc (partial search algorithm)). The random sampling with replacement (bootstraping) + The set of homogeneous machine learning algorithms(ensemble learning). Since there is no dependency between estimators, it can be executed in parallel.

# Data Sampling
**Data Sampling** is a statistical analysis technique to select, manipulate, and analyze a representative subset of data points → identify patterns + trends in the larger dataset being examined. Reducing the biases via sampling is very important, since data in the world is full of potential biases.

(Guide: [Guide 1](https://towardsdatascience.com/5-probabilistic-training-data-sampling-methods-in-machine-learning-460f2d6ffd9), [Guide 2](https://www.techtarget.com/searchbusinessanalytics/definition/data-sampling))
## Data vs Dataset
| Data | Dataset |
| ---- | ------- |
| Infinite | Finite |
| Unfixed | Fixed |
| Dynamic | Stationary |

## Probabilistic Training Data Sampling Methods
### 1. Simple Random Sampling
All the samples in the population have the same chance of being sampled.

* Pros: The method is straightforward, easy to implement.
* Cons: Rare classes in the population might not be sampled in the selection.

### 2. Stratified Sampling
![000stratified](https://user-images.githubusercontent.com/48712088/204133745-c471adfc-8774-446b-9e4e-b8c1b0e7835f.jpg)
Divide the population into several groups according to the requirements, and sample from each group separately.

* Pros: The sampled subsets are ensured to contain classes.
* Cons: The population is not always dividable. For example, it will be very hard to use this method in a multi-label learning task.

### 3. Cluster Sampling
![000cluster](https://user-images.githubusercontent.com/48712088/204136872-fb47048d-8f7a-4ed7-9d98-9318500e122d.jpg)
Divide the population into subsets based on a defined factor, and do random sampling on clusters.

* Pros: Since it selects only certain groups, it requires fewer resources (Generally cheaper), The division into homogeneous groups → Feasibility of the sampling ↑ + More subjects included in the study.
* Cons: Biased samples, Sampling error ↑
