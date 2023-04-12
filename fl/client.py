import flwr as fl
from models.cv_models import loading_checkpoint
from training.strategy import training_model
from utils.evaluation import evaluation
from config.configloader import data_cfg, strategy_cfg, model_cfg
from fl.clientdata import load_clients
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


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, global_model, x, y, xt, yt):
        self.cid = cid
        self.model = global_model
        self.x = x
        self.y = y
        self.xt = xt
        self.yt = yt
        self.history = None

    def get_parameters(self, config):
        print('[Client {}] get_parameters'.format(self.cid))
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model on the client's data given parameters."""
        print('[Client {}] fit, config: {}'.format(self.cid, config))
        self.model.set_weights(parameters)
        if warm_start != str(0):
            print('[Client {}] Continue the training by loading the checkpoint from epoch {}'
                  .format(self.cid, warm_start))
            loading_checkpoint(self.model, model_name, warm_start)

        print('[Client {}] Start training the model'.format(self.cid))
        local_model, history = training_model(self.x, self.y, self.model, model_name, nb_epoch,
                                              warmstart='0', batch_size=batch_size, generator_type=generator_type,
                                              augmentation=augmentation, class_weight=class_weight,
                                              vali_ratio=vali_ratio, input_shape=input_shape)
        self.history = history
        self.model = local_model
        return local_model.get_weights(), len(self.y), {'loss': history.history['loss']}

    def evaluate(self, parameters, config):
        """Evaluate model on the client's data given parameters."""
        print('[Client {}] evaluate, config: {}'.format(self.cid, config))
        self.model.set_weights(parameters)
        print('[Client {}] Evaluating the model on the test set and store everything in {}'
              .format(self.cid, models_dir))
        evaluation(self.xt, self.yt, self.model, self.history, os.path.join(models_dir, model_name))


def get_client_model(client_id):
    clientmodels = load_clients()
    print('Returning client {}'.format(client_id))
    return clientmodels[client_id]


def client_fn(cid) -> fl.client.NumPyClient:
    print('Creating [Client {}]'.format(cid))
    global_model, x, y, xt, yt = get_client_model(cid)
    return FlowerClient(cid, global_model, x, y, xt, yt)
