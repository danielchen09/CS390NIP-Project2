import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
from Model import *
from NeuralNets import *
from Dataset import *
from Config import *
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# random.seed(1618)
# np.random.seed(1618)
# tf.random.set_seed(1618)

# ALGORITHM = "guesser"
# ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"
# ALGORITHM = "efnetb0"

# DATASET = "mnist_d"
# DATASET = "mnist_f"
# DATASET = "cifar_10"
# DATASET = "cifar_100_c"
DATASET = "cifar_100_f"
dataset = Dataset()

config = DEFAULT_CONFIG


def set_dataset(ds=None):
    global DATASET
    global dataset
    global config

    if ds is None:
        ds = DATASET
    if ds == "mnist_d":
        dataset = DigitMNIST()
    elif ds == "mnist_f":
        dataset = FashionMNIST()
    elif ds == "cifar_10":
        dataset = CIFAR10()
    elif ds == "cifar_100_f":
        dataset = CIFAR100F()
        config = cifar100f_config()
    elif ds == "cifar_100_c":
        dataset = CIFAR100C()

    print(f'Dataset: {ds}')
    return ds


# =========================<Pipeline Functions>==================================


def train_model(dataset: Dataset):
    x_train, y_train = dataset.get_training_data()
    if ALGORITHM == "guesser":
        print("Building and training Guesser.")
        return GuesserModel(dataset.flatten_shape, dataset.num_classes).train(x_train)
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return ANNModel(dataset.flatten_shape, dataset.num_classes, config).train(x_train, y_train)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return CNNModel(dataset.input_shape, dataset.num_classes, config).train(x_train, y_train)
    elif ALGORITHM == "efnetb0":
        print("Building and training EfficientNetB0.")
        return EfficientNet(dataset.input_shape, dataset.num_classes, config).train(x_train, y_train)
    else:
        raise ValueError("Algorithm not recognized.")


def run_model(data, model):
    return model.predict(data)


def eval_results(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    return accuracy


# =========================<Main>================================================


def main():
    ds = set_dataset()
    model = train_model(dataset)
    preds = run_model(dataset.get_test_data()[0], model)
    acc = eval_results(dataset.get_test_data(), preds)
    log(ALGORITHM, ds, acc, config)


def test():
    p = Plotter.from_log()
    p.bar('Combine_Accuracy_Plot', save='Combine_Accuracy_Plot.pdf')


if __name__ == '__main__':
    main()
