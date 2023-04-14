from utils.visualization import creating_dir, plot_hist, plot_confmat, make_gradcam_heatmap, make_gradcam_img, \
    get_img_array, view_img
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
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


def evaluation(model, model_name, xt, yt, filedir):
    creating_dir(filedir)
    y_prediction = model.predict(xt)
    training_history = read_history_files(model_name)
    plot_hist(training_history, os.path.join(filedir, 'training_history'))
    confmat = confusion_matrix(yt, y_prediction.argmax(axis=1))
    plot_confmat(confmat, os.path.join(filedir, 'confusion_matrix'))
    report = classification_report(yt, y_prediction.argmax(axis=1),
                                   target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.to_csv(os.path.join(filedir, 'evaluation_report_test.csv'))
    model.save(os.path.join(filedir, 'final.model'))
    n_samples = 10
    sampled_img_paths = np.random.choice(yt, n_samples)
    last_conv_layer_name = 'top_conv'  # Conv_1
    gradcam_img_arrs, labels = [], []

    for img_ar in sampled_img_paths:
        img_arr = get_img_array(xt[img_ar])
        heatmap_arr, label = make_gradcam_heatmap(img_arr, model, last_conv_layer_name)
        gradcam_img_arr = make_gradcam_img(
            xt[img_ar],
            heatmap_arr,
            # cam_path=os.path.join(proc_data_path, img_path.split(os.path.sep)[-1]),
        )
        gradcam_img_arrs.append(gradcam_img_arr)
        labels.append(label.numpy())
    view_img(gradcam_img_arrs, n_samples, filedir)
