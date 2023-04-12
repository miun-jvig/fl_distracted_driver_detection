from preprocessing.data import read_hdf5
from config.configloader import client_cfg, data_cfg, model_cfg
from models.cv_models import create_model, compiling
from pathlib import Path

# data
nb_clients = int(client_cfg['nb_clients'])
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


def load_clients():
    print('Reading the dataset from {}'.format(hdf5_dir))
    train_files = []
    test_files = []
    train_data = {}
    test_data = {}

    for i in range(nb_clients):
        # loading the different datasets (train, validation and test)
        train_file = read_hdf5(hdf5_dir, f'train-{i}', rows, cols)
        test_file = read_hdf5(hdf5_dir, f'test-{i}', rows, cols)
        train_files.append(train_file)
        test_files.append(test_file)

    for i, file in enumerate(train_files):
        train_data[f'x{i}'] = file['/images']
        train_data[f'y{i}'] = file['/meta']

    for i, file in enumerate(test_files):
        test_data[f'xt{i}'] = file['/images']
        test_data[f'yt{i}'] = file['/meta']

    # clients
    clientmodels = {}
    for i in range(nb_clients):
        clientmodels[str(i)] = (model, train_data[f'x{i}'], train_data[f'y{i}'],
                                test_data[f'xt{i}'], test_data[f'yt{i}'])
    return clientmodels
