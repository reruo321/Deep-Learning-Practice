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
**Kernel Method** is a classification algorithm, and "Kernel SVM" is the most famous thing.
**SVM(Support Vector Machine)** is a supervised method used for linear regression and classification problems. It tries to make the best decision boundary in the middle of the void, so that it seperates data points into two classes.

![001decision](https://user-images.githubusercontent.com/48712088/204344699-a420297d-747d-4284-a375-19026181bf48.jpg)

While SVM performs linear classification, the kernel method allows the algorithm to fit the maximum-margin hyperplane in a transformed feature space, therefore it can also efficiently perform a non-linear classification.

![001SVM](https://user-images.githubusercontent.com/48712088/204521791-7cea8193-45d8-4730-a2eb-10ad6c8c735d.jpg)

It seems good to mapping data to high-dimensional space for making the classification problem easier, but it is hard to realize it with computer actually. Here is why the kernel method is needed. It does not need to get the actual coordinate of mapped data in higher dimension. It just require to get the distance between two data points, which is efficiently calculated by "kernel function".

![001kernel](https://user-images.githubusercontent.com/48712088/205038180-5f0f0a44-68f7-4cdc-8c9e-bdeb8af0316b.jpg)

### Decision Tree
**Decision Tree** has a structure similar to flowchart, and it classifies input data points or predicts output value.
It is easy to visualize and understand.

#### Random Forest
**Random Forest** is an algorithm based on decision tree.
