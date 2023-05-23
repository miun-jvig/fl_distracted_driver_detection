from preprocessing.data import read_hdf5
from config.configloader import data_cfg
from pathlib import Path

# data
nb_h5_train_files = int(data_cfg['nb_h5_train_files'])
rows, cols = int(data_cfg['rows']), int(data_cfg['cols'])
hdf5_dir = Path(data_cfg['data_dir'])


def load_test_data():
    """Load test data from a .h5 file"""
    # loading the test dataset
    file_test = read_hdf5(hdf5_dir, 'test', rows, cols)
    xt = file_test["/images"]
    yt = file_test["/meta"]
    return xt, yt


def load_training_data():
    """
    Load training data from one or more .h5 files.

    Returns:
        A dictionary containing train data and labels. The key for the dictionary is '0', '1', ..., 'n', which works
        well as start_simulation creates clients with cid with similar numbers, i.e. '0', '1', ..., 'n'.
    """
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
