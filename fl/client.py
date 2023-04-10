import flwr as fl
from models.cv_models import create_model, compiling, loading_checkpoint
from training.strategy import training_model
from preprocessing.data import read_hdf5
from pathlib import Path
from utils.evaluation import evaluation
from configparser import ConfigParser
import os

config_file = 'config.ini'
print('Reading {} as the configuration file'.format(config_file))
config_object = ConfigParser(comment_prefixes=('#', ';'))
config_object.read(config_file)

data_cfg = config_object['DATA']
server_cfg = config_object['SERVER']
model_cfg = config_object['MODEL']
strategy_cfg = config_object['STRATEGY']

rows, cols = int(data_cfg['rows']), int(data_cfg['cols'])
hdf5_dir = Path(data_cfg['data_dir'])
input_shape = (rows, cols, 3)

server_address = server_cfg['address']
fl_nb_rounds = server_cfg['num_rounds']

model_name = model_cfg['model_name']
init = model_cfg['init']
trainable = model_cfg.getboolean('trainable')
fc_layers = list(map(int, eval(model_cfg['fc_layers'])))
classes = int(model_cfg['classes'])
nb_epoch = int(model_cfg['nb_epoch'])
warm_start = model_cfg['warm_start']
batch_size = int(model_cfg['batch_size'])
vali_ratio = float(model_cfg['vali_ratio'])
models_dir = model_cfg['models_dir']
class_weight = model_cfg.getboolean('class_weight')

augmentation = strategy_cfg.getboolean('augmentation')
generator_type = strategy_cfg['generator_type']

# loading the different datasets (train, validation and test)
print('Reading the dataset from {}'.format(hdf5_dir))
file_train = read_hdf5(hdf5_dir, 'train', rows, cols)
file_test = read_hdf5(hdf5_dir, 'test', rows, cols)

x = file_train["/images"]
y = file_train["/meta"]
file_train.close()

xt = file_test["/images"]
yt = file_test["/meta"]
file_test.close()

print('Creating a {} model'.format(model_name))
model = create_model(model_name, input_shape, classes, fc_layers, trainable, init)
compiling(model)
# model.summary()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, x_train, y_train):
        self.cid = cid
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.history = None

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model on the client's data given parameters."""
        if warm_start != str(0):
            print('Continue the training by loading the checkpoint from epoch {}'.format(warm_start))
            loading_checkpoint(self.model, model_name, warm_start)

        print('Start training the model')
        self.model.set_weights(parameters)
        local_model, history = training_model(self.x_train, self.y_train, self.model, model_name, nb_epoch,
                                              warmstart='0', batch_size=batch_size, generator_type=generator_type,
                                              augmentation=augmentation, class_weight=class_weight,
                                              vali_ratio=vali_ratio,
                                              input_shape=input_shape)
        self.history = history
        self.model = local_model
        return local_model

    def evaluate(self, parameters, config):
        """Evaluate model on the client's data given parameters."""
        self.model.set_weights(parameters)
        print('Evaluating the model on the test set and store everything in {}'.format(models_dir))
        evaluation(xt, yt, self.model, self.history, os.path.join(models_dir, model_name))


def client_fn(cid) -> fl.client.NumPyClient:
    return FlowerClient(cid, x, y)
