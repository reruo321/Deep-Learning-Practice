# Mathematical Component of Neural Network
## The First Meeting with Neural Network
The MNIST dataset is a well-known large database of handwritten digits, and is included in Keras as NumPy format.

    # Loading the MNIST dataset in Keras
    
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load.data()
