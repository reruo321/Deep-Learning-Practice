# Mathematical Component of Neural Network
From this chapter, I will add some sample codes on Google Colab notebooks.

## The First Meeting with Neural Network
### The MNIST Dataset
The MNIST dataset is a well-known large database of handwritten digits, and is included in Keras as NumPy format.

### Neural Network
#### Layer
**Layer** is the core of neural network, which is a kind of data processing filter. It exports more meaningful representation from input data. Most of deep learning models consist of some connected simple layers, and they gradually refines data.

#### Compiling
To train a neural network, there are three things to be included while compiling it:

  1. **Loss Function**: Measures the performance of neural network with training data, so that it can be trained on the right track.
  2. **Optimizer**: Mechanism to update the network based on input data + loss function.
  3. **Indicator to Monitor Training and Testing**: An example is accuracy, the ratio of correctly classified images.

## Data Expression for Neural Network
### Tensor
**Tensor** is a multidimensional NumPy array to save data. Most of the latest machine learning systems uses tensor as basic data structure. It is a generalized form of array with the arbitrary number of dimensions, and it usually contains numerical data.

#### Scalar Tensor
**Scalar tensor(Scalar, zero-dimensional tensor, 0D tensor, array scalar)** is a tensor containing only one number. It is either float32 or float64 in NumPy. You can check the number of dimensions (or axes) of a NumPy array, by using *ndim*. Scalar tensor is ndim == 0. Notice that the number of axes is also called "rank".

    # Scalar Tensor
    >>> import numpy as np
    >>> x = np.array(12)
    >>> x
      array(12)
    >>> x.ndim
      0

#### Vector (1D Tensor)
**Vector**, or **1D tensor** is an array of numbers, which has only one axis.

    # 1D Tensor
    # This example vector is 5D vector.
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> x
      array([1, 2, 3, 4, 5])
    >>> x.ndim
      1

While 5D vector has one axis with 5 dimensions, 5D tensor has five axes. The better expression for 5D tensor will be "five rank tensor".

## The Gear of Neural Network: Tensor Operator

## 
