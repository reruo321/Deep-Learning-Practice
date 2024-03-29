# Unit 3 - Initializing Neural Network
## 3.1 Structure of Neural Network

![001weight](https://user-images.githubusercontent.com/48712088/215152063-4d1717fc-11b7-4e8a-a273-b6f407ab9360.jpg)

As we saw in the previous units, there are some relevant elements for training a neural network.

* **Layers** that form **network (or model)**
* **Input data** and its **target**
* **Loss function** which defines feedback signal for learning
* **Optimizer** that decides learning method.

### Layer: The Component of Deep Learning
**Layer** is the core data structure of neural network. It is a data processing module that imports/exports one or more tensors as a input/output. Most layers have a state called weight. **Weight** is one or more tensors learned by SGD, containing knowledge the network has learnt.

Each layer has different suitable tensor format and data processing method. A Keras deep learning model is made by constructing data conversion pipeline, which is a connection of compatible layers. **Layer compatibility** means that each layer only accepts specific size of input/output.

### Model: The Network of Layers
Deep learning is a **Directed Acyclic Graph(DAG)** being made with layers. 

We will see these things frequently while studying:

* A network with two branches
* A network with multiple outputs
* An inception block

Network structure defines **hypothesis space**. In other words, we can limit the space to a series of specific tensor operations, which maps input data to output one, by selecting specific network structure.

### Loss Function & Optimizer: The Key to Adjust Learning Process
After defining the network structure, we should select two more things:

* **Loss function (Objective function)**: Should be minimized while training.
* **Optimizer**: Decides how to update the network, based on loss function. Embodies specific kind of SGD.

A network with multiple output can have multiple loss functions, one per each output. However, since SGD sets only one scalar loss value as base, all loss will be calculated the average and combined to one scalar in the network with multiple output.

### Selecting Object Function
It is very important to select appropriate object function for a problem. We will study general choices in detail later.

(Examples)

* Classification problem with two classes: **Binary crossentropy**
* Classification problem with multiple classes: **Categorical crossentropy**
* Regression problem: **Mean squared error**
* Sequence learning problem: **CTC(Connection Temporal Classification)**

## 3.2 Keras Introduction
In this book we use [Keras](https://keras.io) for code examples. It is a deep learning framework for Python which allows users to make and train almost kinds of deep learning models easily.

The characteristics of Keras are:

* Can be run both in CPU and GPU, with the same code.
* Has easy-to-use API so that can easily make deep learning model prototype.
* Supports convolutional neural network (for computer vision) and recurrent neural network (for sequence processing). Can combine them.
* Can make any network structure: From General Adversarial Network(GAN) to Neural Turing Machine, and etc....

### Keras, Tensorflow, Theano, and CNTK
**Keras** is a model-level library that provides high-level elements for making deep learning models. It does not treat low-level operations such as tensor operation or differentiation, but instead uses tensor library which is provided by Keras backend engine. Keras doe not depend on just one tensor library. It has module structure, and several backend engines are in sync with it.

The current Keras can use Tensorflow, Theano, and CNTK(Microsoft Cognitive Toolkit) as a backend engine, and more engines will be available later.

### Development with Keras: Quick Guide
Typical Keras process is similar to what we saw from the MNIST example.

1. Define training data, which includes input tensor + target tensor.
2. Define network (or model) that has layer mapping the input + target.
3. Set learning process by selecting loss function, optimizer, and measurement metrics to monitor.
4. Repeatably call model's *fit()* method for training data.

There are two ways to define a model: use ***Sequential* class**, the network which stacks layers sequentially, or use **Functional API**, creating DAG that is able to make completely random structure.

Whichever way you chose while defining the model, all the following steps are the same. Step 3. Learning process is set on compiling step.

Last, pass NumPy array of input data to the *fit()* method in the model.

## 3.3 Deep Learning Computer Setting
It is preferred to use GPU than CPU, but there is also an option to use AWS EC2 GPU instance or Google Cloud platform. Unix is the best OS to select, so Windows users will get benefit by setting dual booting with Ubuntu. Tensorflow, CNTK, or Theano must be installed to use Keras. (This book will use Tensorflow mainly.)

### Jupyter Notebook: The Optimized Way for Deep Learning Test
**Jupyter Notebook** is an application for web browsers which supports text format with template, and enables users to run Python codes. It can divide and run long codes into small ones, so that it allows users to perform interactive programming.

### Starting Keras: Two Ways
It is recommended to use one of two ways below:

1. Use [The Official Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/) to run Keras examples on EC2, via Jupyter Notebook. It is good if your local computer does not have any GPU.
2. Install all things on local UNIX computer, and run Jupyter Notebooks or normal Python scripts. Do this if your computer have high-end NVIDIA GPU card.

### Pros and Cons on Cloud Deep Learning Process
One of the easiest cloud service to start deep learning is AWS EC2.
#### Pros
* Simple and cheap way for deep learning, if there's no good GPU card.
* No difference between cloud and local, if you use Jupyter Notebook
#### Cons
* Not good for long-term grand-scale deep learning process: using the cloud instance will become expensive.

### What GPU Card for Deep Learning?
Try to pick out NVIDIA graphic cards...

## 3.4 Movie Review Classification: Binary Classification Example
We will learn on **binary classification (or two-class classification)**.

### IMDB Dataset
**IMDB dataset** is a dataset whose data are from the Internet Movie Database. It has 25,000 training data and 25,000 testing data. 50% of the data are positive, and the rest are negative.

Very important thing: YOU MUST NOT TRAIN AND TEST A MACHINE LEARNING MODEL **WITH THE SAME DATA!** We should take a look at the model's capability to predict the target on the new data, not the training data it would already know well.

### Preparing Data
We cannot inject number list into neural network directly. However, we can do it by converting the list into tensor. There are two ways to do it.

* Add **padding** to list so that it has the same length → convert it into **integer tensor** with the shape (samples, sequence_length). Use the layer which can process it as the **1st layer** in the neural network. (**Embedding layer**)
* **One-hot encoding** the list to convert it into vector with 0 and 1.

### Creating Neural Network Model
The example input data is vector, and the label is scalar. (0 or 1) It may be one of the simplest problem we would see. Anyway, the network that would be nice for this kind of problems is fully connected layer(FCL) using *relu* activation function.

    # This is the fully connected layer using relu.
    Dense(16, activation='relu')
    
An argument *16* above is the number of hidden units. One hidden unit becomes one dimension in the representation space that layer shows.
Also, we embodied *Dense* layer by connecting this tensor operation.

    output = relu(dot(W, input) + b)
    
If there are 16 hidden units, it means that the shape of the weight array *W* is (input_dimension, 16). Dot product of input data and *W* makes the input projected onto the 16-dimensional space. After that, bias vector *b* is added, and *relu* operation is also applied to it.

If hidden units ↑,
* Pros: Neural network can study more complex representation
* Cons: Calculation cost ↑, Can study unwanted pattern (it only enhances performance on training data, not on testing one)

There are two important things to decide when you stack *Dense* layers.

* How many layers to use?
* How many hidden units to put on each layer?

We have used the activation function *relu* in the middle hidden layer in the example. It makes all negative numbers into 0. Next, the last layer uses *signmoid* function which compresses all numbers into the number in \[0, 1\]. It indicate that a sample with high possibility of target '1' has the higher chance of positive review.
