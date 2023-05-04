from models.cv_models import create_model, compiling
from config.configloader import model_cfg, data_cfg
import tensorflow as tf

# data
model_name = model_cfg['model_name']
rows, cols = int(data_cfg['rows']), int(data_cfg['cols'])
input_shape = (rows, cols, 3)
init = model_cfg['init']
trainable = model_cfg.getboolean('trainable')
fc_layers = list(map(int, eval(model_cfg['fc_layers'])))
classes = int(model_cfg['classes'])

# creating global model
model = create_model(model_name, input_shape, classes, fc_layers, trainable, init)
compiling(model, finetuning=trainable)


def load_model():
    return model


def create_converter(old_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(old_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter


# def representative_data_gen():
#     for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
#         yield [input_value]


# def create_lite_model_int(old_model):
#     converter = create_converter(old_model)
#     converter.representative_dataset = representative_data_gen
#     # Ensure that if any ops can't be quantized, the converter throws an error
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#     # Set the input and output tensors to uint8 (APIs added in r2.3)
#     converter.inference_input_type = tf.uint8
#     converter.inference_output_type = tf.uint8
#     lite_model = converter.convert()
#     return lite_model


def create_lite_model(old_model):
    converter = create_converter(old_model)
    lite_model = converter.convert()
    return lite_model
