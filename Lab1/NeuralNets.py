from Model import Model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from Util import *
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers.experimental import preprocessing
from DataGenerator import *
import efficientnet


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(0.1),
    ]
)

class ANNModel(Model):
    def __init__(self, input_size: int, output_size: int, config=None):
        super(ANNModel, self).__init__(input_size, output_size, config)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, x_train, y_train) -> Model:
        self.model.fit(x_train, y_train, epochs=30, verbose=1)
        return self

    def predict(self, x):
        return to_one_hot(self.model.predict(x), self.output_size)


class CNNModel(Model):
    def __init__(self, input_size: tuple, output_size: int, config=None):
        super(CNNModel, self).__init__(input_size, output_size, config)
        regularizer = None
        if self.config['use_regularization']:
            if self.config['regularization'] == 'l2':
                regularizer = tf.keras.regularizers.l2(self.config['lambda'])
            elif self.config['regularization'] == 'l1':
                regularizer = tf.keras.regularizers.l1(self.config['lambda'])

        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=input_size, padding='same',
                                   kernel_regularizer=regularizer
                                   )
        )
        for _ in range(self.config['64_layers'] - 1):
            self.model.add(
                tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                                       kernel_regularizer=regularizer
                                       ),
            )
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(tf.keras.layers.BatchNormalization())
        if self.config['use_dropout']:
            self.model.add(tf.keras.layers.Dropout(self.config['dropout']))
        for _ in range(self.config['128_layers']):
            self.model.add(
                tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same',
                                       kernel_regularizer=regularizer
                                       )
            )
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(tf.keras.layers.BatchNormalization())
        if self.config['use_dropout']:
            self.model.add(tf.keras.layers.Dropout(self.config['dropout']))
        for _ in range(self.config['256_layers']):
            self.model.add(
                tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',
                                       kernel_regularizer=regularizer
                                       )
            )
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(tf.keras.layers.BatchNormalization())
        if self.config['use_dropout']:
            self.model.add(tf.keras.layers.Dropout(self.config['dropout']))

        self.model.add(tf.keras.layers.Flatten())
        for l in self.config['linear_layers']:
            self.model.add(tf.keras.layers.Dense(l, activation='relu'))
        self.model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, x_train, y_train) -> Model:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1618)
        callback = []
        if self.config['auto_lr']:
            callback += [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)]
        if self.config['save']:
            callback += [tf.keras.callbacks.ModelCheckpoint(filepath='CheckPoints/efnetb0.ckpt',
                                                            save_weights_only=True,
                                                            verbose=1)]
        if self.config['augment']:
            self.model.fit(datagen.flow(x_train, y_train),
                           epochs=self.config['epochs'],
                           batch_size=self.config['batch_size'],
                           callbacks=callback,
                           validation_data=(x_val, y_val))
        else:
            self.model.fit(x_train, y_train,
                           epochs=self.config['epochs'],
                           batch_size=self.config['batch_size'],
                           callbacks=callback,
                           validation_data=(x_val, y_val))
        self.model.save_weights('cnn_weight')
        return self

    def predict(self, x):
        return to_one_hot(self.model.predict(x), self.output_size)


class EfficientNet(Model):
    def __init__(self, input_size, output_size, config):
        super(EfficientNet, self).__init__(input_size, output_size, config)
        self.model = tf.keras.models.Sequential()
        efnb0 = tf.keras.applications.EfficientNetB1(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            classes=output_size
        )
        self.model.add(efnb0)
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

        self.model.summary()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, _x_train, _y_train):
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=123)

        for train_index, val_index in sss.split(_x_train, _y_train):
            x_train, x_val = _x_train[train_index], _x_train[val_index]
            y_train, y_val = _y_train[train_index], _y_train[val_index]

        print('-')
        print(f'x train shape: {x_train.shape}')
        print(f'x val shape: {x_val.shape}')
        print(x_train[0])
        print(x_val[0])
        train_generator = DataGenerator(x_train, y_train, augment=True)
        val_generator = DataGenerator(x_val, y_val, augment=False)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=10,
            restore_best_weights=True)
        auto_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=1)
        save_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath='CheckPoints/efnetb0.ckpt',
                                                       save_weights_only=True,
                                                       verbose=1)
        self.model.fit(train_generator,
                       validation_data=val_generator,
                       callbacks=[early_stopping, auto_lr, save_ckpt],
                       verbose=1,
                       epochs=self.config['epochs'])
        self.model.save_weights('efnetb0_weight')

    def predict(self, x):
        return to_one_hot(self.model.predict(DataGenerator(x, mode='predict', augment=False, shuffle=False)),
            self.output_size)
