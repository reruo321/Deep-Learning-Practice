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

# Probabilistic Training Data Sampling Methods
