from utils.visualization import creating_dir, plot_hist, plot_confmat
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os


def read_history_files(model_name):
    history_dir = "./logs/" + model_name
    history_files = glob.glob(history_dir + "/client-*/history-*.csv")
    histories = {}
    for file_path in history_files:
        cid = file_path.split("\\")[-2].split("-")[-1]
        history = pd.read_csv(file_path)
        history_dict = {}
        for metric_name in history.columns:
            history_dict[metric_name] = list(history[metric_name])
        histories[cid] = history_dict
    return histories


def evaluate_compressed_model(model, xt, use_lite_model):
    if 'float' in use_lite_model:
        use_lite_model = "float32"  # this value is used with xt.astype(use_lite_model), and float8 doesn't exist
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # update input shape based on the actual shape of xt
    input_shape = input_details[0]['shape']
    input_shape[0] = xt.shape[0]
    interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
    interpreter.allocate_tensors()
    xt = xt.astype(use_lite_model)
    interpreter.set_tensor(input_details[0]['index'], xt)
    interpreter.invoke()
    y_prediction = interpreter.get_tensor(output_details[0]['index'])
    return y_prediction


def evaluate_keras_model(model, xt, filedir):
    y_prediction = model.predict(xt)
    model.save(os.path.join(filedir, 'final.model'))
    return y_prediction


def evaluation(model, model_name, xt, yt, filedir, use_lite_model=None):
    creating_dir(filedir)
    if 'int' or 'float' in use_lite_model:
        y_prediction = evaluate_compressed_model(model, xt, use_lite_model)
    else:
        y_prediction = evaluate_keras_model(model, xt, filedir)
    confmat = confusion_matrix(yt, y_prediction.argmax(axis=1))
    training_history = read_history_files(model_name)
    plot_hist(training_history, os.path.join(filedir, 'training_history'))
    plot_confmat(confmat, os.path.join(filedir, 'confusion_matrix'))
    report = classification_report(yt, y_prediction.argmax(axis=1),
                                   target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.to_csv(os.path.join(filedir, 'evaluation_report_test.csv'))
