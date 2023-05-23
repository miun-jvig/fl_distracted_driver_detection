import flwr as fl
from models.cv_models import loading_checkpoint
from training.strategy import training_model
from config.configloader import data_cfg, strategy_cfg, model_cfg
from fl.clientdata import load_training_data, load_test_data
from models.model import load_model
from training.utils import preprocess_labels
import pandas as pd
import numpy as np
import random

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
    """Client used for federated learning training and local evaluation"""
    def __init__(self, cid, global_model, x, y, xt, yt):
        self.cid = cid
        self.model = global_model
        self.x = x
        self.y = y
        self.xt = xt
        self.yt = yt

    def get_parameters(self, config):
        print('[Client {}] get_parameters'.format(self.cid))
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model on the client's data given parameters."""
        print('[Client {}] fit'.format(self.cid))
        self.model.set_weights(parameters)
        if warm_start != str(0):
            print('[Client {}] Continue the training by loading the checkpoint from epoch {}'
                  .format(self.cid, warm_start))
            loading_checkpoint(self.model, self.cid, model_name, warm_start)
        print('[Client {}] Start training the model'.format(self.cid))
        local_model, history = training_model(self.x, self.y, self.model, model_name, nb_epoch, self.cid,
                                              warmstart='0', batch_size=batch_size, generator_type=generator_type,
                                              augmentation=augmentation, class_weight=class_weight,
                                              vali_ratio=vali_ratio, input_shape=input_shape)
        filepath = f"./logs/{model_name}/client-{self.cid}/history-{self.cid}.csv"
        save_history(history, config['server_round'], filepath)
        self.model = local_model
        evaluation_metrics = {'loss': history.history['loss'], 'accuracy': history.history['accuracy']}
        return self.model.get_weights(), len(self.y), evaluation_metrics

    def evaluate(self, parameters, config):
        """Evaluate model on the client's data given parameters."""
        print('[Client {}] evaluate, config: {}'.format(self.cid, config))
        self.model.set_weights(parameters)
        yt = preprocess_labels(self.yt, len(np.unique(self.yt)))
        loss, accuracy = self.model.evaluate(self.xt, yt)
        return loss, len(self.yt), {'accuracy': float(accuracy)}


def save_history(history, round_num, filepath):
    """Saves history of training round to a .csv file, later used for creating training/loss history graphs"""
    hist_df = pd.DataFrame(history.history)
    hist_df['round'] = round_num
    if round_num == 1:
        with open(filepath, mode='w') as f:
            hist_df.to_csv(f, index=False)
    else:
        with open(filepath, mode='a') as f:
            hist_df.to_csv(f, header=False, index=False)


def get_client_train_data(client_id):
    """Get train data using client_id, which is a (cid) in client_fn"""
    clientmodels = load_training_data()
    return clientmodels[client_id]


def get_random_client_train_data():
    """Gets a random .h5 file for training, an improvement to this would be using crossvalidation"""
    clientmodels = load_training_data()
    key, value = random.choice(list(clientmodels.items()))
    print(f"Chose random model {key}")
    return value


def client_fn(cid) -> fl.client.NumPyClient:
    """
    Responsible for creating clients in start_simulation. Note that clients are ephemeral, run on threads,
    and deleted after use.

    Args:
        cid: start_simulation will automatically give a cid to clients created.

    Returns:
        A FlowerClient, as defined above, which can train and evaluate data.
    """
    print('Creating [Client {}]'.format(cid))
    # x, y = get_client_train_data(cid)
    x, y = get_random_client_train_data()
    xt, yt = load_test_data()
    global_model = load_model()
    return FlowerClient(cid, global_model, x, y, xt, yt)
