import flwr as fl
from models.cv_models import loading_checkpoint, create_model, compiling
from training.strategy import training_model
from utils.evaluation import evaluation
from pathlib import Path
from config.configloader import data_cfg, strategy_cfg, model_cfg
from preprocessing.data import read_hdf5
import os

# data from config
rows, cols = int(data_cfg['rows']), int(data_cfg['cols'])
input_shape = (rows, cols, 3)
model_name = model_cfg['model_name']
nb_epoch = int(model_cfg['nb_epoch'])
warm_start = model_cfg['warm_start']
batch_size = int(model_cfg['batch_size'])
vali_ratio = float(model_cfg['vali_ratio'])
models_dir = model_cfg['models_dir']
class_weight = model_cfg.getboolean('class_weight')
augmentation = strategy_cfg.getboolean('augmentation')
generator_type = strategy_cfg['generator_type']

# data
hdf5_dir = Path(data_cfg['data_dir'])
rows, cols = int(data_cfg['rows']), int(data_cfg['cols'])
init = model_cfg['init']
trainable = model_cfg.getboolean('trainable')
fc_layers = list(map(int, eval(model_cfg['fc_layers'])))
classes = int(model_cfg['classes'])

# loading the different datasets (train, validation and test)
print('Reading the dataset from {}'.format(hdf5_dir))
file_train = read_hdf5(hdf5_dir, 'train', rows, cols)
file_test = read_hdf5(hdf5_dir, 'test', rows, cols)

# train/test data
x = file_train["/images"]
y = file_train["/meta"]
xt = file_test["/images"]
yt = file_test["/meta"]
file_test.close()
file_train.close()

# creating global model
print('Creating a {} model'.format(model_name))
model = create_model(model_name, input_shape, classes, fc_layers, trainable, init)
compiling(model)
# model.summary()

# clients
client_0 = (model, x, y, xt, yt)  # client 1 with its unique dataset
client_1 = (model, x, y, xt, yt)  # client 2 with its unique dataset
clientmodels = {'0': client_0, '1': client_1}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, x, y, xt, yt):
        self.cid = cid
        self.model = model
        self.x = x
        self.y = y
        self.xt = xt
        self.yt = yt
        self.history = None

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model on the client's data given parameters."""
        self.model.set_weights(parameters)
        if warm_start != str(0):
            print('Continue the training by loading the checkpoint from epoch {}'.format(warm_start))
            loading_checkpoint(self.model, model_name, warm_start)

        print('Start training the model')
        local_model, history = training_model(self.x, self.y, self.model, model_name, nb_epoch,
                                              warmstart='0', batch_size=batch_size, generator_type=generator_type,
                                              augmentation=augmentation, class_weight=class_weight,
                                              vali_ratio=vali_ratio, input_shape=input_shape)
        self.history = history
        self.model = local_model
        return local_model

    def evaluate(self, parameters, config):
        """Evaluate model on the client's data given parameters."""
        self.model.set_weights(parameters)
        print('Evaluating the model on the test set and store everything in {}'.format(models_dir))
        evaluation(self.xt, self.yt, self.model, self.history, os.path.join(models_dir, model_name))


def get_client_model(client_id):
    return clientmodels[client_id]


def client_fn(cid) -> fl.client.NumPyClient:
    print("Creating client {}".format(cid))
    model, x, y, xt, yt = get_client_model(cid)
    return FlowerClient(cid, model, x, y, xt, yt)
