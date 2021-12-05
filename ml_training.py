import lzma
import random
from typing import Dict, List

import cv2
from mnist import MNIST
import tensorflow as tf
import numpy as np

from time_util import timeit

reshape = True
epoch = 20
batch_size = 128

level: Dict[str, List[int]] = {
    'light': [2, 4],
    'basic': [4, 8],
    'normal': [8, 16],
    'heavy': [16, 32],
    'too-heavy': [64, 128],
    'stupid': [128, 256],
    'insane': [256, 512],
}


def print_sample(sample: np.ndarray, **kwargs):
    print(f"+{'-' * 28}+")
    for y in range(28):
        ss = ""
        for x in range(28):
            v = sample[y * 28 + x]
            # print(v, sep=None)
            if v == 0:
                s = ' '
            elif v < 50:
                s = '.'
            elif v < 128:
                s = 'o'
            elif v < 200:
                s = "X"
            else:
                s = "W"
            ss += s
        print(f"|{ss}|")
    print(f"+{'-' * 28}+")
    if kwargs is not None and 'num' in kwargs and kwargs.get("save_it") is True:
        cv2.imwrite(f"sample-{kwargs['num']}.png", np.reshape(sample, (28, 28)))


class MNIST_lzma(MNIST):
    def __init__(self, path='.', mode='vanilla', return_type='lists', gz=False, lzma=False):
        super().__init__(path, mode, return_type, gz)
        self.lzma = lzma

    def opener(self, path_fn, *args, **kwargs):
        if self.lzma:
            return lzma.open(path_fn + '.lzma', *args, **kwargs)
        else:
            return super().opener(path_fn, *args, **kwargs)


@timeit
def load_train(**kwargs):
    print("Loading train data")
    mndata = MNIST_lzma('./MNIST/', return_type='numpy', lzma=True)

    train, tr_lbl = mndata.load_training()
    rnd = int(len(tr_lbl) * random.random())
    print(f"This is a {tr_lbl[rnd]}")
    print_sample(train[rnd], save_it=False, num=tr_lbl[rnd])
    return train, tr_lbl


@timeit
def load_test(**kwargs):
    print("Loading test data")
    mndata = MNIST_lzma('./MNIST/', return_type='numpy', lzma=True)

    test, te_lbl = mndata.load_testing()
    rnd = int(len(te_lbl) * random.random())
    print(f"This is a {te_lbl[rnd]}")
    print_sample(test[rnd], save_it=False, num=te_lbl[rnd])
    return test, te_lbl


def to_dataset(images, labels):
    ds = []
    for i, lbl in enumerate(labels):
        if reshape:
            image = np.reshape(images[i], (28, 28, 1))
        else:
            image = images[i]
        ds.append((image, int(lbl)))

    return tf.data.Dataset.from_generator(lambda: ds, (tf.int32, tf.int32))


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    print(f"label: {label}")
    return tf.cast(image, tf.float32) / 255., label


@timeit
def create_train(**kwargs):
    train, tr_lbl = load_train(**kwargs)
    ds_train = to_dataset(train, tr_lbl).map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(len(tr_lbl)) #ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    print(len(list(ds_train.as_numpy_iterator())))
    return ds_train


@timeit
def create_test(**kwargs):
    test, te_lbl = load_test(**kwargs)
    ds_test = to_dataset(test, te_lbl).map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_test


@timeit
def train(train, test, level_name: str, **kwargs):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(level[level_name][0], (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
        tf.keras.layers.Conv2D(level[level_name][1], (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),

        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    if not reshape:
        callbacks = []
    else:
        plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=0,
            mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            './checkpoints', monitor='val_loss', verbose=0, save_best_only=False,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None
        )
        stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=0,
            mode='auto', baseline=None, restore_best_weights=True
        )

        callbacks = [plateau, checkpoint]  # , stopping]

    history = model.fit(
        train,
        epochs=epoch,
        validation_data=test,
        callbacks=callbacks,
        verbose=2
    )
    model.save("./auto")
    return history, model


def do_it(level_name: str, **kwargs):
    ds_train = create_train(**kwargs)
    ds_test = create_test(**kwargs)

    return train(ds_train, ds_test, level_name, **kwargs)


if __name__ == "__main__":
    h, m = do_it('basic')
