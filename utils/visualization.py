import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import os


def creating_dir(filedir):
    os.makedirs(filedir, exist_ok=True)


def create_plot(ax, x, y, y_label, x_label, title, labels, fontsize=12):
    ax.plot(x, 'b', y, 'r')
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(labels, fontsize=fontsize, loc='best')


def plot_hist(training_history, filename):
    for i, client_history in training_history.items():
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
        # first plot
        create_plot(ax1, client_history['accuracy'], client_history['val_accuracy'], 'Accuracy Rate', 'Iteration',
                    'Categorical Cross Entropy (Data augmentation)', ['Training Accuracy', 'Validation Accuracy'])

        # second plot
        create_plot(ax2, client_history['loss'], client_history['val_loss'], 'Loss', 'Iteration', 'Learning Curve',
                    ['Training Loss', 'Validation Loss'])

        # save figure
        nb_epochs = len(client_history['accuracy'])
        fig.suptitle(f"ClientID = {i}, Epochs = {nb_epochs}", fontsize=16)
        fig.tight_layout()
        plt.savefig(filename+f"_client-{i}"+'.png')
        # plt.show()


def plot_confmat(confmatrix, confname, labels=None):
    if labels is None:
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure(figsize=(8, 6))
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    sns.heatmap(confmatrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    # plt.ylim((16,14))
    plt.xlabel('Predicted label')
    plt.savefig(confname+'.png')
    #plt.show()


def get_img_array(img_array):
    array = np.expand_dims(img_array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index


def make_gradcam_img(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((128,128))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    # superimposed_img.save(cam_path)
    return superimposed_img


def view_img(img_arrs, n_samples, filedir):
    n_cols = 5
    n_rows = n_samples // n_cols if n_samples % n_cols == 0 else n_samples // n_cols + 1
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
    for i in range(n_samples):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, xticks=[], yticks=[])
        ax.imshow(img_arrs[i])
        ax.axis("off")
    plt.savefig(os.path.join(filedir,'gradcam_img_examples.png'))