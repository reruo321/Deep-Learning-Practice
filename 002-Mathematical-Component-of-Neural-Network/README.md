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

As the number of dimensions (axes) of a tensor, the rank of tensor, increases, it is called as 0D tensor (Scalar) → 1D tensor (Vector) → 2D tensor (Matrix) → 3D tensor → ... Note that **XD tensor** means the tensor has X axes, while **XD vector** has an axis with X dimensions. (X elements) The better technical expression for XD tensor would be "X rank tensor", though.

There are three main attributes that defines tensor. Let the example tensor is:

    x = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15]])

1. Number of Axis (Rank): You can check the number of dimensions (or axes) of a NumPy array, by using *ndim*. As the example is a 2D tensor, it has two axes. 
2. Shape: It is a tuple in Python, which represents the number of dimensions for each axis of a tensor. The shape of the example tensor is (3, 5).
3. Data Type: The data type of data included in a tensor. It can be float32, uint8, float64, rarely char, or etc.. Since tensor should be allocated in contiguous memory in advance, NumPy array and many other libraries do not support variable length string.

#### Scalar Tensor
**Scalar tensor(Scalar, zero-dimensional tensor, 0D tensor, array scalar)** is a tensor containing only one number. It is either float32 type or float64 type in NumPy. Scalar tensor is ndim == 0. It has no shape.

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
    # This example is a 5D vector, not a 5D tensor
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> x
      array([1, 2, 3, 4, 5])
    >>> x.ndim
      1

#### Matrix (2D Tensor)
**Matrix**, or **2D tensor** has two axes: row and column.

If a matrix x is:

    x = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15]])

The first row of it is \[1, 2, 3, 4, 5\], and the first column of it is \[1, 6, 11\].

#### High-Dimensional Tensor
We can make a ND tensor by combining (N-1)D tensors. Usually tensors from 0D to 4D are used in deep learning, and sometimes also 5D for processing video data.

## The Gear of Neural Network: Tensor Operator

## 
