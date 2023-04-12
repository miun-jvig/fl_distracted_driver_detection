import tensorflow as tf
from keras.utils import to_categorical
from sklearn.utils import class_weight
import numpy as np


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)


def train_val_split(x,y, val_ratio=0.1):
    pivot = x.shape[0] - round(x.shape[0]*val_ratio)
    return x[:pivot], y[:pivot], x[pivot:], y[pivot:]


def preprocess_labels(y, classes):
    return to_categorical(np.array(y).astype(int), num_classes=classes)


def compute_class_weight(y):
   return dict(zip(np.unique(y),
                            class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)))
