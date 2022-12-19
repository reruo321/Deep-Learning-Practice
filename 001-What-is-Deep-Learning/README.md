# Unit 1 - What is Deep Learning?
## Artifical Intelligence vs Machine Learning vs Deep Learning
![001AI](https://user-images.githubusercontent.com/48712088/204151390-2e189e8b-1dd3-405f-b2fd-cec9d6b6f1f0.png)

### 1. Artificial Intelligence
**Artificial Intelligence** is the theory + development of computer systems to perform like human intelligence. The early AI was started as symbolic AI, which explicitly define knowledge and rules for the behavior of computer programs.

### 2. Machine Learning
**Machine Learning** gives computers the ability to learn, without being explicitly programmed. To solve more complicated problems like image classification, it became a substitute for symbolic AI.

* Categories:
  1. **Supervised Learning**: Example input + Desired output is given by a human teacher → Learn a general rule that maps inputs to outputs.
  2. **Unsupervised Learning**: No labels are given → Find structure in its input by itself.
  3. **Reinforcement Learning**: Interacts with a dynamic environment (Ex. driving a vehicle or playing a PvP game) → Tries to maximize reward feedback.

### 3. Deep Learning
**Deep Learning** is machine learning algorithms with brain-like logical structure of algorithms(= Artificial Neural Networks).
Since it uses multiple layers in the network, it is also called as "layered representations learning" or "hierarchical representations learning".

## Mechanism
![001weight](https://user-images.githubusercontent.com/48712088/202853688-a912b524-b96a-4b06-b53f-af35fcdcc51c.jpg)

## History of Machine Learning
### Probabilistic Modeling
**Probabilistic Modeling** is an application of statistical principles to data analysis. Some examples are Naive Bayes Algorithm, Logistic Regression, and Multilayer Perceptrons.

### Early Version of Neural Network
In 1950s, the main idea of neural network was found, but it took a dozen years to find way to train large neural network.
In 1980s, people found several backpropagation algorithms, and applied them to neural network.
LeNet is the first (convolutional) neural network with backpropagation that performs image classification on handwritten zip codes.

### Kernel Method
**Kernel Method** is a classification algorithm, and [Kernel SVM](https://github.com/reruo321/Deep-Learning-Practice/blob/main/000-Appendix/README.md#svm) is the most famous thing.

### Decision Tree
**Decision Tree** has a structure similar to flowchart, and it classifies input data points or predicts output value.
It is easy to visualize and understand.

#### Random Forest
**Random Forest** is an algorithm based on decision tree. It consists of a lot of different decision trees, so that it operates as an ensemble. It was the most preferable algorithm in the early years of Kaggle.

#### Gradient Boosting Machine
**Gradient Boosting Machine** is an ensemble algorithm using weak prediction models, which are typically decision trees. It combines weak learners into a single strong learner in an iterative fashion. Not only it usually outperforms random forest, it is very powerful method that is most used in Kaggle, except deep learning.

### Neural Network Again
In 2010, neural network did not attract most science communities' attention. However, the tables have been turned since deep neural network trained by GPU won an academic image classification competition 2011. Since 2012 when ConvNet (deep convolutional neural network) won ImageNet challenge with an amazing feat, ConvNet has been the main algorithm in most computer vision works.
Moreover, as deep learning has been used in various problems, it becomes a substitution of SVM or decision tree.

### Characteristic of Deep Learning
Deep learning runs better in many problems, and it completely automates feature engineering, the key of machine learning.

Old machine learning methods, which are shallow learning, just converts input data into one or two continuous layers of representations, with SVM or decision tree. However, these methods generally cannot provide well-refined representation on complicated problems. Therefore, human should convert initial input data in several ways, so that these machine learning methods process it well, which is called "feature engineering". **Feature Engineering** is the process of using domain knowledge to extract features from raw data, to operate machine learning algorithm using those features.

By the way, deep learning can completely automate this step by learning all features at once. Therefore, we can use one end-to-end deep learning model as a substitute for high-degree of process. Besides, it creates more complicated representations throughout layers, and gradual intermediate representations (IR) are jointly learned with it. Thus, it is way more powerful than former machine learning apporaches, which learn all representation layers continuously. (greedily)

| Table | Old ML Methods | Deep Learning |
| ----- | ------- | ------- |
| Type | Shallow Learning | Deep Learning |
| Representation | One or two continuous layers | Complicated through layers |
| Learning | Continuously (Greedily) | Simultaneously |
| Feature Engineering | Manual | Automated |

Deep Learning Summary:

* Completely automates feature learning (the key of machine learning) by learning all features at once.
* Enables to use one end-to-end deep learning model, instead of high-degree of process.
* Creates more complicated representations through layers.
* Gradual intermediate representations are jointly learned with deep learning.

### Recent Trends of Deep Learning
To get recent information of machine learning, it is good to check Kaggle to look for lectures and competitions. Especially, focus on gradient boosting machine (using XGBoost library) for shallow learning, and deep learning (using Keras library) for perceptual problems.

## Why Deep Learning? Why Now?
There were three technical powers that made progress in machine learning:

1. Hardware
2. Dataset & Benchmark
3. Advancement of Algorithm

### 1. Hardware
* Process of Cluster Development: With huge amount of CPUs → With small amount of GPUs → With hundreds of GPUs → Developing deep learning chip, TPU ...

As GPU which is more powerful than CPU in parallelism has been developed, computing of a cluster with multiple GPUs also has been rapidly grown.

#### CPU vs GPU

| Table | CPU | GPU |
| ----- | --- | --- |
| Usage | General Purposes | Creating Images for Computer Graphics </br> & Video Game Consoles </br> Accelerate Massive Repetitive Calculations |
| Parallelism | △ | ○ |
| Video Rendering | △ | ○ |
| Complex Mathematical Calculations | △ | ○ |
| Cryptocurrency Mining | △ | ○ |
| Instruction Sets | ○ | △ |

### 2. Dataset & Benchmark
Data is fuel of deep learning. Large number of data on the Internet and big datasets such as ImageNet have led to growth of deep learning.

### 3. Advancement of Algorithm
It was a challenge to find a stable way to train deep neural network. The biggest problem was gradient propagation, since feedback signal to train neural network faded away as layers increased.

Deep learning has been highlighted after three simple but important algorithms improved:

1. Activation Function - Better suited to neural network layers
2. Weight Initialization - Made pretraning per layer be out of use
3. Optimization - RMSProp, Adam, ...

Moreover, high level techniques such as batch normalization, residual connection, and depthwise separable convolution were developed.

## Revolution of AI
The three reasons that the current state of deep learning is called "revolution" are:

1. Simplicity - No need to use feature engineering. Can make a model with a few Tensor operation and train it with end-to-end
2. Expandability - Deep learning is easily parallelized in GPU or TPU, so takes advantage of Moore's Law.
3. Utility & Reusability - Unlike many other previous machine learning methods, deep learning model can be trained by additional data, without starting from the beginning. Enable to use continual online learning. Can reuse a model to other purpose, or some work to make stronger model. Can also apply deep learning model on a small dataset.

