# fl_distracted_driver_detection
Based on work done during the Research Project INSECTT (https://www.insectt.eu). Initially cloned from https://github.com/davbuf/distracted_driver_detection, which is a centralized ML solution to detect distracted drivers, with the goal to extend the work to a federated learning setting.

# Challenge 
Originally, this code was cloned from a previous centralized ML solution that was used to detect distracted drivers with an approximate 90 % accuracy. To do this, this project used labeled training and test data from the State Farm Distracted Driver dataset (https://www.kaggle.com/c/state-farm-distracted-driver-detection), which is a collection of images used for ML. The codebase is created with a pretrained model (of the users choice) and can be tuned by using different strategies such as: data augmentation, sampling techniques, mixup and classweighting. To extend the solution to a federated learning setting, the project used the federated learning framework flower (https://flower.dev).

# Requisites
The project requires that you have the images from the State Farm Dataset, and several different Python libraries. The most important part of these are:

- TensorFlow 2.9.1 (due to a bug with TensorFlow, https://discuss.tensorflow.org/t/using-efficientnetb0-and-save-model-will-result-unable-to-serialize-2-0896919-2-1128857-2-1081853-to-json-unrecognized-type-class-tensorflow-python-framework-ops-eagertensor/12518/20)
- Due to having TensorFlow 2.9.1, we also need Protobuf 3.20.0 or lower
- Patience of a saint, the training takes a long time

# Instructions
## 1 Data conversion 
Convert the dataset using the python script in preprocessing/data.py. It will find all the images from the Statefarm dataset, resize them and store them in a H5 file(s) for training. Do the same for the test set. 

## 2 Model experiments
Use the configuration file (config.ini) to select the dataset, the model and the training strategy to experiment. Then run the main script. It should train a model on the data. The logs are automatically stored into a created logs directory. At the end of the training, an evaluation step is done on the test set and the results are stored in the 'models_dir' (see config.ini)
