from preprocessing.data import read_hdf5
from config.configloader import data_cfg, model_cfg
from models.cv_models import create_model, compiling
from pathlib import Path
import pandas as pd

# data
nb_h5_train_files = int(data_cfg['nb_h5_train_files'])
rows, cols = int(data_cfg['rows']), int(data_cfg['cols'])
input_shape = (rows, cols, 3)
hdf5_dir = Path(data_cfg['data_dir'])
model_name = model_cfg['model_name']
init = model_cfg['init']
trainable = model_cfg.getboolean('trainable')
fc_layers = list(map(int, eval(model_cfg['fc_layers'])))
classes = int(model_cfg['classes'])

# creating global model
model = create_model(model_name, input_shape, classes, fc_layers, trainable, init)
compiling(model)
# model.summary()


def load_model():
    return model


def save_history(history, round_num, filepath):
    hist_df = pd.DataFrame(history.history)
    hist_df['round'] = round_num
    if round_num == 1:
        with open(filepath, mode='w') as f:
            hist_df.to_csv(f, index=False)
    else:
        with open(filepath, mode='a') as f:
            hist_df.to_csv(f, header=False, index=False)


def load_test_data():
    # loading the test dataset
    file_test = read_hdf5(hdf5_dir, 'test', rows, cols)
    xt = file_test["/images"]
    yt = file_test["/meta"]
    return xt, yt


def load_training_data():
    print('Reading the dataset from {}'.format(hdf5_dir))
    train_files = []
    train_data = {}

    for i in range(nb_h5_train_files):
        # loading the train dataset
        train_file = read_hdf5(hdf5_dir, f'train-{i}', rows, cols)
        train_files.append(train_file)

    for i, file in enumerate(train_files):
        train_data[f'x{i}'] = file['/images']
        train_data[f'y{i}'] = file['/meta']

    # clients
    clientmodels = {}
    for i in range(nb_h5_train_files):
        clientmodels[str(i)] = (train_data[f'x{i}'], train_data[f'y{i}'])
    return clientmodels
