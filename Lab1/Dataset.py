import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class Dataset:
    def __init__(self):
        self.num_classes = 0
        self.input_shape = np.array([0])
        self.flatten_shape = 0

    def _get_data(self, dataset):
        (x_train, y_train), (x_test, y_test) = dataset
        x_train = x_train.reshape((x_train.shape[0], *self.input_shape))
        x_train = x_train / 255
        x_test = x_test.reshape((x_test.shape[0], *self.input_shape))
        x_test = x_test / 255
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        return (x_train, y_train), (x_test, y_test)

    def get_data(self):
        pass

    def get_training_data(self):
        return self.get_data()[0]

    def get_test_data(self):
        return self.get_data()[1]

    def summary(self):
        print(f'input dimensions: {self.input_shape}')
        print(f'output dimensions: {self.num_classes}')
        print(f'train set size: {self.train_size}')
        print(f'test set size: {self.test_size}')


class MNIST(Dataset):
    def __init__(self):
        self.num_classes = 10
        self.input_shape = (28, 28, 1)
        self.flatten_shape = np.prod(self.input_shape)


class DigitMNIST(MNIST):
    def get_data(self):
        return self._get_data(tf.keras.datasets.mnist.load_data())


class FashionMNIST(MNIST):
    def get_data(self):
        return self._get_data(tf.keras.datasets.fashion_mnist.load_data())


class CIFAR(Dataset):
    def __init__(self):
        self.num_classes = 10
        self.input_shape = (32, 32, 3)
        self.flatten_shape = np.prod(self.input_shape)


class CIFAR10(CIFAR):
    def get_data(self):
        return self._get_data(tf.keras.datasets.cifar10.load_data())


class CIFAR100F(CIFAR):
    def __init__(self):
        super(CIFAR100F, self).__init__()
        self.num_classes = 100

    def get_data(self):
        return self._get_data(tf.keras.datasets.cifar100.load_data(label_mode='fine'))


class CIFAR100C(CIFAR):
    def __init__(self):
        super(CIFAR100C, self).__init__()
        self.num_classes = 20

    def get_data(self):
        return self._get_data(tf.keras.datasets.cifar100.load_data(label_mode='coarse'))

