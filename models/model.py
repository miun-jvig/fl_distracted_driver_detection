from models.cv_models import create_model, compiling
from config.configloader import model_cfg, data_cfg
from preprocessing.data import read_hdf5
from pathlib import Path
import tensorflow as tf
import numpy as np

# data
model_name = model_cfg['model_name']
rows, cols = int(data_cfg['rows']), int(data_cfg['cols'])
input_shape = (rows, cols, 3)
init = model_cfg['init']
trainable = model_cfg.getboolean('trainable')
fc_layers = list(map(int, eval(model_cfg['fc_layers'])))
classes = int(model_cfg['classes'])
hdf5_dir = Path(data_cfg['data_dir'])

# creating global model
model = create_model(model_name, input_shape, classes, fc_layers, trainable, init)
compiling(model, finetuning=trainable)


def load_model():
    return model


def summary(lite_model):
    return tf.lite.experimental.Analyzer.analyze(model_content=lite_model)


def representative_data_gen():
    file_unlabel = read_hdf5(hdf5_dir, 'unlabeled', rows, cols)
    train_images = np.array(file_unlabel['/images'], dtype=np.float32)
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]


def create_lite_model(old_model, lite_model_type):
    converter = tf.lite.TFLiteConverter.from_keras_model(old_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if lite_model_type == 'int':
        converter.representative_dataset = representative_data_gen
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    lite_model = converter.convert()
    return lite_model
