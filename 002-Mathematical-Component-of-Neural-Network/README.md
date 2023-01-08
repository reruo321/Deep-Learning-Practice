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
We can make a ND tensor by combining (N-1)D tensors. Usually tensors from 0D to 4D are used in deep learning, and sometimes also up to 5D for processing video data.

### Manipulate Tensor with NumPy
**Slicing** is a behavior selecting specific elements in an array. You can slice some numbers from the previous example by calling

    # Slicing a Number
    train_images[i]
    
    # Slicing 11th, 12th, ..., 100th number
    my_slice = train_images[10:100]
    train_images[10:100, :, :]
    # Slice train_images.shape == (60000, 28, 28) into shape (90, 28, 28), same as the above ones
    my_slice = train_images[10:100, 0:28, 0:28]
    
    # To select bottom right 14x14 pixels from the image
    my_slice = train_images[:, 14:, 14:]
    
    # To select the middle 14x14 pixels from the image
    my_slice = train_images[:, 7:-7, 7:-7]

### Batch Data
Generally, the first axis (since the index starts from 0, it would be actually the 0th axis) of all data tensor in deep learning is sample axis. (sample dimension) Deep learning model does not process all dataset at once, but divide data into several small batches.

In MNIST numeral data, the n-th batch with the size 128 is

    batch = train_images[128 * n : 128 * (n+1)]

### Tensor in the Real World
1. Vector Data
2. Sequence Data
3. Image Data
4. Video Data

#### 1. Vector Data
**Vector data** is 2D tensor which shape is (samples, features).

Most of tensors are vector data, and in those datasets one data point can be encoded to vector, so that batch data would be encoded to 2D tensor. (= array of vector)
Sample axis is the first, and feature axis is the second.

Example: If a dataset contains information of 100,000 people on age, zip code, and salary - It can be stored in a tensor with the shape (100000, 3).

#### 2. Sequence Data
**Sequence data(Time series data)** is a dataset which includes time axis and be stored in a 3D tensor.

As each sample is encoded to the sequence of vectors (2D tensors), the batch data would be encoded to 3D tensor.

#### 3. Image Data
**Image data** is usually 3D; height, width, and color channel. Although a black-and-white image can be stored in a 2D tensor because of its only one color channel, (its dimension size of color channel is 1.) it is conventially stored in a 3D tensor.

128 batches for a 256x256 black-and-white image can be stored in a tensor with the shape == (128, 256, 256, 1). Meanwhile, 128 batches for a 256x256 color image can be stored in a tensor with the shape == (128, 256, 256, 3).

There are two types of ways that assign the size of image tensors. Google's Tensorflow machine learning framework uses channel-last, (samples, height, width, color_depth). On the other hand, Theano puts the depth of color channel right behind the batch axis, (samples, color_depth, height, width).

#### 4. Video Data
**Video data** is one of the few data that needs 5D tensor in real-life. As a frame can be stored in 3D tensor with its shape (height, width, color_depth), 4D tensor with its shape (frames, height, width, color_depth) can store a sequence of frames. Therefore, batches for multiple videos would be stored in 5D tensor whose shape == (samples, frames, height, width, color_depth).

## The Gear of Neural Network: Tensor Operation
Some kinds of **tensor operations** being applied to mathematical data tensor, such as tensor addition and tensor multiplication, can express all conversions that a deep neural network has learned.

The first example made a neural network by stacking Dense layers.

    keras.layers.Dense(512, activation='relu')
    
This layer gets 2D tensor as an input and also an output.

    output = relu(dot(W, input) + b)
    
Note that W is a 2D tensor, b is a vector. Both are the properties of the layer. The operation has three tensor operations: dot, addition, and ReLU.

※ relu(x) = max(x, 0). If an input x is bigger than 0, it returns x, otherwise 0.

### Element-Wise Operation
ReLU and addition are element-wise operations. **Element-wise operation** is applied to each element in a tensor independently, meaning that it can do a high-degree of parallel implementation.

### Broadcasting
If there is an operation that needs two tensors with the same shape, but if the size of operands are different, broadcasting would be adjust the smaller tensor. It is very inefficient thing that a new tensor is made. Fortunately, broadcasting is a virtual operation, not making any new tensor.

How broadcasting works:

1. A broadcasting axis is added to smaller tensor, to fit *ndim* of bigger tensor.
2. Repeat (1) to fit the size of smaller tensor to the size of bigger one.

#### Example
Let X's shape == (32, 10), t's shape == (10,).

1. Add the first axis to t so that its shape becomes (1, 10).
2. Repeat t 32 times, then tensor T becomes shape == (32, 10).
3. Now we can do a tensor addition between X and T.

## 
