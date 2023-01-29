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
