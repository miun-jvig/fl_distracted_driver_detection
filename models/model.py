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
    """Loads model"""
    return model


def summary(lite_model):
    """Returns the summary of a TensorFlow Lite model"""
    return tf.lite.experimental.Analyzer.analyze(model_content=lite_model)


def representative_data_gen():
    """
    This function is needed for TensorFlow Lite if you wish to do post-training integer quantization, as stated in
    https://www.tensorflow.org/lite/performance/model_optimization. It is used to send unlabeled representative
    samples which is read from a .h5 file containing unlabeled data.
    """
    file_unlabel = read_hdf5(hdf5_dir, 'unlabeled', rows, cols)
    train_images = np.array(file_unlabel['/images'], dtype=np.float32)
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]


def create_lite_model(old_model, lite_model_type):
    """
    Creates a TensorFlow Lite model from the old model, and then optimizes the precision rate of the model. Default is
    float32, can create float16, uint8, and float8.

    Args:
        old_model: The model you wish to quantize.
        lite_model_type: What value, i.e. float16, uint8, float8.

    Returns:
        A quantized model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(old_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if 'int' in lite_model_type:
        converter.representative_dataset = representative_data_gen
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif lite_model_type == 'float16':
        converter.target_spec.supported_types = [tf.float16]

    lite_model = converter.convert()
    return lite_model
