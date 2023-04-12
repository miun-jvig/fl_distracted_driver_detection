from .visualization import creating_dir, plot_hist, plot_confmat, make_gradcam_heatmap, make_gradcam_img, get_img_array, view_img
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os


def evaluation(xtest, ytest, model, history, filedir):
    creating_dir(filedir)
    y_prediction = model.predict(xtest)
    plot_hist(history, os.path.join(filedir, 'training_history'))
    confmat = confusion_matrix(ytest, y_prediction.argmax(axis=1))
    plot_confmat(confmat, os.path.join(filedir, 'confusion_matrix'))
    report = classification_report(ytest, y_prediction.argmax(axis=1),
                                target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.to_csv(os.path.join(filedir, 'evaluation_report_test.csv'))
    model.save(os.path.join(filedir, 'final.model'))
    n_samples = 10
    sampled_img_paths = np.random.choice(ytest, n_samples)
    last_conv_layer_name = 'top_conv' #Conv_1
    gradcam_img_arrs, labels = [], []

    for img_ar in sampled_img_paths:
        img_arr = get_img_array(xtest[img_ar])
        heatmap_arr, label = make_gradcam_heatmap(img_arr, model, last_conv_layer_name)
        gradcam_img_arr = make_gradcam_img(
            xtest[img_ar],
            heatmap_arr,
            #cam_path=os.path.join(proc_data_path, img_path.split(os.path.sep)[-1]),
        )
        gradcam_img_arrs.append(gradcam_img_arr)
        labels.append(label.numpy())
    view_img(gradcam_img_arrs, n_samples, filedir)
