import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from os.path import exists


def to_one_hot(output, n_classes):
    return to_categorical(np.argmax(output, axis=1), n_classes)


class Plotter:
    def __init__(self, labels, save_dir='Plots', x_data=None, y_data=None, width=0.25):
        self.save_dir = save_dir
        self.labels = labels
        if not isinstance(labels, list):
            self.labels = [labels]

        self.x_data = x_data
        if x_data is None:
            self.x_data = []

        if y_data is None:
            self.y_data = {}
        elif isinstance(y_data, list):
            if len(self.labels) != len(y_data):
                raise AttributeError('len(x) != len(y)')
            self.y_data = {}
            for i in range(len(self.labels)):
                self.y_data[self.labels[i]] = y_data[i]
        elif isinstance(y_data, dict):
            self.y_data = y_data

        self.width = width
        self.color = ('r', 'g', 'b', 'c', 'm', 'y')

    def add_x_data(self, x):
        self.x_data.append(x)

    def add_y_data(self, x, y):
        self.y_data[x] = y

    def bar(self, title, save=None):
        n = len(self.x_data)
        ind = np.arange(n)
        for i in range(len(self.labels)):
            plt.bar(ind + i * self.width,
                    self.y_data[self.labels[i]],
                    width=self.width,
                    label=self.labels[i],
                    color=self.color[i])
        plt.xticks(ind + self.width / n, tuple(self.x_data))
        plt.legend(loc='best')
        plt.title(title)
        if save is not None:
            plt.savefig(f'{self.save_dir}/{save}', dpi=100)
        plt.show()

    @staticmethod
    def from_log(algorithm=None):
        log_data = read_log()
        labels = set()
        x = set()
        for key, value in log_data.items():
            x.add(key[1])
            labels.add(key[0])
        x = list(x)
        labels = list(labels)
        data = np.zeros((len(labels), len(x)))
        for key, value in log_data.items():
            data[labels.index(key[0])][x.index(key[1])] = value[0]
        data = data.tolist()

        if algorithm is not None:
            data = [data[labels.index(algorithm)]]
            labels = [algorithm]
            print(x, data)
            x = [x for _, x in sorted(zip(data[0], x), reverse=True)]
            data = [sorted(data[0], reverse=True)]
            print(x, data)

        p = Plotter(list(labels), x_data=x, y_data=data)
        return p


def read_log():
    log_data = {}
    if not exists('Logs.txt'):
        return log_data
    with open('Logs.txt', 'r') as f:
        for line in f.readlines():
            line_data = line.split('|')
            line_data[3] = line_data[3].replace('\n', '')
            log_data[(line_data[0], line_data[1])] = (float(line_data[2]), line_data[3])
    return log_data


def log(algorithm, dataset, accuracy, info=None):
    if info is None:
        info = {}
    info = str(info)

    if not exists('Logs.txt'):
        with open('Logs.txt', 'w'):
            pass

    log_data = read_log()
    if (algorithm, dataset) not in log_data or \
            accuracy > log_data[(algorithm, dataset)][0]:
        log_data[(algorithm, dataset)] = (float(accuracy), info)

    with open('Logs.txt', 'w') as f:
        for key, value in log_data.items():
            f.write(f'{key[0]}|{key[1]}|{value[0]}|{value[1]}\n')


