from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.tensorflow import balanced_batch_generator
import tensorflow as tf
from keras.utils import Sequence


class BalancedDataGenerator(Sequence):
    """
    ImageDataGenerator + RandomOverSampling
    or
    ImageDataGenerator + RandomUnderSampling
    x: tensor of examples
    y: tensor of labels
    datagen: ImageDataAnnotator generator
    batch_size: size of the batch
    undersampling: boolean, True for balancing a batch using Undersampling and
                False for using Oversampling.
    """
    def __init__(self, x, y, datagen, batch_size=32, undersampling=True):
        self.datagen = datagen
        self.batch_size = min(batch_size, x.shape[0])
        datagen.fit(x)
        if undersampling:
            self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y,
                                                                      sampler=RandomUnderSampler(),
                                                                      batch_size=self.batch_size, keep_sparse=True)

        else:
            self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y,
                                                                  sampler=RandomOverSampler(),
                                                                  batch_size=self.batch_size, keep_sparse=True)
        self._shape = (self.steps_per_epoch * batch_size, *x.shape[1:])

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()