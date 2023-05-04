from preprocessing.data import read_hdf5
from config.configloader import data_cfg
from pathlib import Path

# data
nb_h5_train_files = int(data_cfg['nb_h5_train_files'])
rows, cols = int(data_cfg['rows']), int(data_cfg['cols'])
hdf5_dir = Path(data_cfg['data_dir'])


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
