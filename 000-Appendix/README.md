# Statistical Inference
## Naive Bayes
**Naive Bayes classifier** is one of simple probabilistic classifiers based on applying Bayes' theorem with *strong (naive) independence assumptions between the features*. Bias ↑, Variance ↓.

(Easy Guide: https://www.youtube.com/watch?v=O2L2Uv9pdDA)

### Bayes' Theorem
**Bayes' theorem** describes the probability of an event, based on prior knowledge of conditions that might be related to the event.

#### Prior Probability
**Prior probability** is the probability distribution that would express one's belief, in Bayesian statistical inference.
#### Likelihood
**Likelihood** is the probability of something discrete and individual. (excluding something continuous like weight or height)

## Bias & Variance

### 1. Bias
**Bias** is the inability for a machine learning method to capture the true relationship.

### 2. Variance
**Variance** is the amount by which the prediction would change, if we fit the model to a different training data set.

#### Consequence of Variance
**Consequence of Variance** is the difference in fits between data sets.

### A. Overfitting
**Overfitting** is the production of an analysis too fitting to a particular set of data → may fail to fit to additional data or predict future observations reliably. When a complicatied model too fits to a training data, it is likely to have a higher rate of error on new unseen data.

### B. Underfitting
**Underfitting** occurs when a mathematical model cannot adequately capture the underlying structure of the data.

### Ensemble Learning
**Ensemble Learning** uses multiple learning algorithms → Obtain better predictive performance than just using one algorithm. Some examples are *Regularization*, *Boosting*, and *Bagging*.

#### 1. Regularization
**Regularization** is a process that changes the result answer to be "simpler": To obtain results for ill-posed problems OR to prevent overfitting.
