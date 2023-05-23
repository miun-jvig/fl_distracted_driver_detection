from flwr.common import Metrics
from fl.clientdata import load_test_data
from models.model import load_model
from fl.client import client_fn
from config.configloader import client_cfg, config_file, model_cfg
from typing import Dict, List, Optional, Tuple
from training.utils import preprocess_labels
import flwr as fl
import numpy as np
import torch

# data from config
model_name = model_cfg['model_name']
nb_clients = int(client_cfg['nb_clients'])
nb_rounds = int(client_cfg['nb_rounds'])
device = client_cfg['device']
nb_device = int(client_cfg['nb_device'])
params = load_model().get_weights()
print('Reading {} as the configuration file'.format(config_file))
print('Creating a {} model'.format(model_name))


def fit_config(server_round: int):
    """
    Return training configuration for each round. This function is used to send information to the clients, as they
    are run on threads. The clients can reach this information with the help of the parameter "config" in
    get_parameters, fit, and evaluate.

    Args:
        server_round: the current federated learning round.

    Returns:
        training configuration for each round.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit results using weighted average."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar], ) \
        -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    """
    Evaluate global model parameters using an evaluation function.

    Saves the weights of the model which is later used in main.py for evaluation.
    """
    # specify path for server checkpoint
    path = "./logs/" + model_name + "/server/"
    filepath = path + f"history.csv"
    checkpoint_path = path + f"/cpft-{server_round}.ckpt"
    model = load_model()

    # save the model checkpoint
    if server_round != 1:
        weights = np.array(parameters, dtype=object)
        model.set_weights(weights)
        model.save_weights(checkpoint_path)

    # evaluate the model and compute the loss and accuracy
    xt, yt = load_test_data()
    ytest = preprocess_labels(yt, len(np.unique(yt)))
    loss, accuracy = model.evaluate(xt, ytest)

    if server_round == 1:
        with open(filepath, mode='w') as f:
            f.write("accuracy, loss, round_number\n")
            f.write("{}, {}, {}\n".format(accuracy, loss, server_round))
    else:
        with open(filepath, mode='a') as f:
            f.write("{}, {}, {}\n".format(accuracy, loss, server_round))

    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


strategy = fl.server.strategy.FedAvg(
    initial_parameters=fl.common.ndarrays_to_parameters(params),
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=evaluate,
)

client_resources = None
DEVICE = torch.device(device)
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": nb_device}


def simulation():
    """
    Simulates a federated learning scenario, clients are created on client_fn and run on threads.

    Note that threads are ephemeral and deleted after use.
    """
    results = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=nb_clients,
        config=fl.server.ServerConfig(num_rounds=nb_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    return results
