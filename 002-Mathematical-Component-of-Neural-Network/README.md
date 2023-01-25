# Unit 2 - Mathematical Component of Neural Network
From this chapter, I will add some sample codes on Google Colab notebooks.

## 2.1 The First Meeting with Neural Network
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

## 2.2 Data Expression for Neural Network
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

## 2.3 The Gear of Neural Network: Tensor Operation
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
If there is an operation that needs two tensors with the same shape, but if the size of operands are different, broadcasting would be adjust the smaller tensor. It is a very inefficient thing that a new tensor is made. Fortunately, broadcasting is a virtual operation, not making any new tensor.

How broadcasting works:

1. A broadcasting axis is added to smaller tensor, to fit *ndim* of bigger tensor.
2. Repeat (1) to fit the size of smaller tensor to the size of bigger one.

#### Example
Let X's shape == (32, 10), t's shape == (10,).

1. Add the first axis to t so that its shape becomes (1, 10).
2. Repeat t 32 times, then tensor T becomes shape == (32, 10).
3. Now we can do a tensor addition between X and T.

### Tensor Product
**Tensor product (Dot operation)** is a very useful tensor operation to combine elements of input tensors.

![002dot](https://user-images.githubusercontent.com/48712088/211220256-0489eaa4-f71a-4748-adf7-dea6985777de.png)

Let tensor x.shape == (a, b), y.shape == (b, c). If x · y = z, z.shape would be (a, c).

p.s. If (a, b, c, d) · (d,) → (a, b, c). If (a, b, c, d) · (d, e) → (a, b, c, e).

### Tensor Reshaping
**Tensor reshaping** rearranges a tensor's rows and columns to fit the specific shape. The number of its elements should be equal to original one.

    train_images = train_images.reshape((60000, 28 * 28))
    
**Transposition** is the special reshaping that is often used, which swaps rows for columns.

### Geometric Analysis of Tensor Operation
All tensor operations can be geometrically analyzed. Also, basic geometric operations such as affine transformation, rotation, and scaling can be expressed as tensor operations.

### Geometric Analysis of Deep Learning
We have learnt that a neural network is the connection of tensor operations, and all of them are the geometrical conversion of input data. The step to unfold a manifold of data as neat formulation would be a metaphor for it.

## 2.4 The Engine of Neural Network: Gradient Based Optimization
We have learnt that each layer in the first neural network example had converted the input data like

    output = relu(dot(W, input) + b)
    # W: weight, kernel
    # b: bias

In the random initialization step, a matrix of weights is filled with small random numbers. Although *relu(dot(W, input) + b)* may have useless representation at first, its weight would be gradually adjusted (trained) based on feedback signal.

Training loop is like:

1. Export training sample *x* and a batch of the target *y*.
2. (Forward pass) Run network using *x*, and find prediction, *y_pred*.
3. Estimate the difference between *y_pred* and *y*, and calculate the network loss for this batch.
4. Update all weights in the network to reduce the loss for the batch.

Since each element in all weight matrices takes costly forward pass twice, the method is very inefficient. Therefore, it is better to find the gradient of network weight's loss, using the fact that all operations in neural network are differentiable.

### What is Derivative?
If a function *f(x) = y* (x, y ∈ R) is continuous and smooth, and if epsilon_x is small enough, we can approximate *f* on a point *p* to a linear function with slope *a*.

    f(x + epsilon_x) = y + epsilon_y
    
    ※ If x→p
    f(x + epsilon_x) = y + a * epsilon_x
    
The slope is called **derivative of *f* on *p***. Since all differentiable function f(x) has its derivative function f'(x), we can use it to find x that minimize f(x).

### The Derivative of Tensor Operation: Gradient
**Gradient** is the derivative of tensor operation.

Suppose that there is an input vector *x*, a matrix *W*, a target *y*, and a loss function *loss*. Then we can find the prediction of y, *y_pred*, and also the difference between *y* and *y_pred*.

    y_pred = dot(W, x)
    loss_value = loss(y_pred, y)
    
Also, if *x* and *y* are fixed, we can think this function maps W to the loss value.

    loss_value = f(W)
    
Let's say that the current value of *W* is *W0*. Then we can express the derivative of *f* on *W0* as **gradient(f)(W0)**, the tensor which has the same size as *W*. An element of the tensor *gradient(f)(W0)\[i, j\]* shows the direction and the size of being changed *loss_value*, when W0\[i, j\] is changed. Therefore, *gradient(f)(W0)* is the gradient of *f(W) = loss_value* on *W0*.

We can reduce the value of *f(W)* by moving *W* to the opposite direction of the gradient. (derivative)

### Stochastic Gradient Descent
Theoretically, if a function is differentiable, we can analytically find its minimum value by finding points where the derivative are 0 and comparing them. In other words, we can find the minimum value of a loss function in neural network by solving *gradient(f)(W) = 0*. However, it is actually very difficult to do it since real neural networks often have tens of millions of parameters.

Instead, we can gradually reduce the loss by modifying parameters bit by bit, based on the current loss value in a random batch data. The below is the **mini-batch stochastic gradient descent (mini-batch SGD)** which picks each batch data randomly.

1. Export a training sample batch *x* and its target *y*.
2. Run network using *x*, and find prediction, *y_pred*.
3. Estimate the difference between *y_pred* and *y*, and calculate the network loss for this batch.
4. Find the gradient of the loss function for the network parameter. (backward pass)
5. Move the parameter a bit to the opposite direction of the gradient.

Moreover, it is very important to choose suitable *step* value. Too small value will cause too many repetitions for going down loss function curve, and it is possible to get stuck in local minimum. Meanwhile, if the value is too big, it will move to completely random position in the curve.

Using appropriate size of mini-batch will be a good compromise plan between true SGD (one sample + one target per repetition) and batch SGD. (repetition using all data, accuracy ↑ cost ↑)

There are many SGD variants that consider previously updated weights, such as SGD using momentum, Adagrad, and RMSProp. **Momentum** is the product of the mass and velocity of an object in physics, and its concept is also very important for solving SGD issues of convergence speed and local minimum.

### Derivative Chaining: Backpropagation Algorithm
Suppose that there is a network *f* that contains three tensor operations *a*, *b*, and *c*, and weight matrices *W1*, *W2*, and *W3*.

    f(W1, W2, W3) = a(W1, b(W2, c(W3)))

In calculus, such connected function can be induced by the **chain rule**, *f(g(x))' = f'(g(x)) * g'(x)*. The algorithm that applies the chain rule to gradient calculation in neural network is **backpropagation (reverse-mode automatic differentiation)**. It starts from the final loss value, and it goes backward to find the degree that each parameter contributed to the loss.

Symbolic differentiation helps users not to embody the exact backpropagation algorithm directly. These days, cutting edge frameworks possible to compute symbolic differentiation like Tensorflow will be popular.

## 2.5 Review the First Example

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    
* Data type of input image: float32
* Size of training image: (60000, 784)
* Size of testing image: (10000, 784)

      network = models.Sequential()
      network.add(layers.Dense(512, activation='relu', input_shape(28 * 28,)))
      network.add(layers.Dense(10, activation='softmax'))
      
* Network: 2 Dense layers
* Each layer: includes weight tensor, applies some simple tensor operations to input data
* Weight tensor: where network saves information

      network.compile(optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
                     
* rmsprop: optimizer
* categorical_crossentropy: loss function. used as feedback signal to learn weight tensor, minimized while training.
* Reducing loss: by mini-batch stochastic gradient descent (mini-batch SGD)

      network.fit(train_images, train_labels, epochs=5, batch_size=128)
      
When calling *fit* method, the network repeats training data five times, using mini batches in which each of them has 128 samples. **Epoch** is the repeat on the entire training data. On every epoch, the network calculates the weight gradient for loss in batch, and updates the weight based on the result. Each epoch will perform gradient update 469 times, (60000/128=468.75, the last batch has 96 samples.) so the network will do it total 2,345 times.

## 2.6 Summary
* **Learning**: finding the combination of model parameters that minimizes loss fuction, when training data sample + its target are given.
* **Learning**: randomly select data sample + target's batch → calculate the gradient of parameter of loss. Move the network parameter to the opposite direction of the gradient.
* The fact that neural network is connected with **differentiable tensor operations** makes the whole learning process possible. Uses **the chain rule** to make gradient function which maps the current parameter and batch data.
* Two core concepts we will see in the next unit, **loss** and **optimizer**, should be defined before driving data into the network.
* **Loss**: should be minimized while training → used to measure the success of problem.
* **Optimizer**: defines the exact way that the gradient of loss updates the parameter. Example - RMSProp optimizer, SGD using momentum, etc....
