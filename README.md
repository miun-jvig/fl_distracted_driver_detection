# fl_distracted_driver_detection
Based on work done during the Research Project INSECTT (https://www.insectt.eu). Initially cloned from https://github.com/davbuf/distracted_driver_detection to extend the work from a centralized setting to a federated learning setting.

# Challenge 
Originally this code was done for training a computer vision model to detect distracted drivers on images. A Kaggle challenge, the Statefarm Distracted Driver,
made an available dataset of images (https://www.kaggle.com/c/state-farm-distracted-driver-detection). A codebase was created, containing a collection of computer vision practices where a pretrained model can be tuned by using different strategies such as:
data augmentation, sampling techniques, mixup and classweighting.

To further extend this, the challenge became to extend the codebase to a federated learning setting. This was done using the federated learning framework flower (https://flower.dev).

# Instructions
## 1 Data conversion 
Convert the dataset using the python script in preprocessing/data.py
It will find all the images from the Statefarm dataset, resize them and store them in a H5 file for training. Do the same for the test set. 

## 2 Model experiments
Use the configuration file (config.ini) to select the dataset, the model and the training strategy to experiment. Then run the main script. It should train a model on the data. The logs are automatically stored into a created logs directory. At the end of the training, an evaluation step is done on the test set and the results are stored in the 'models_dir' (see config.ini)

