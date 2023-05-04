from utils.visualization import creating_dir, plot_hist, plot_confmat, make_gradcam_heatmap, make_gradcam_img, \
    get_img_array, view_img
from sklearn.metrics import confusion_matrix, classification_report
from config.configloader import client_cfg
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os

use_lite_model = client_cfg.get('use_lite_model', '').lower() == 'true'


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


def evaluation(model, model_name, xt, yt, filedir):
    creating_dir(filedir)
    if use_lite_model:
        interpreter = tf.lite.Interpreter(model_content=model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # update input shape based on the actual shape of xt
        input_shape = input_details[0]['shape']
        input_shape[0] = xt.shape[0]
        interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
        interpreter.allocate_tensors()
        xt = xt.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], xt)
        interpreter.invoke()
        y_prediction = interpreter.get_tensor(output_details[0]['index'])
    else:
        y_prediction = model.predict(xt)
        model.save(os.path.join(filedir, 'final.model'))
    confmat = confusion_matrix(yt, y_prediction.argmax(axis=1))
    training_history = read_history_files(model_name)
    plot_hist(training_history, os.path.join(filedir, 'training_history'))
    plot_confmat(confmat, os.path.join(filedir, 'confusion_matrix'))
    report = classification_report(yt, y_prediction.argmax(axis=1),
                                   target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.to_csv(os.path.join(filedir, 'evaluation_report_test.csv'))
