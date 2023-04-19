import flwr as fl
import torch
from flwr.common import Metrics
from typing import List, Tuple
from fl.clientdata import model
from fl.client import client_fn
from config.configloader import client_cfg, config_file, model_cfg

# data from config
model_name = model_cfg['model_name']
nb_clients = int(client_cfg['nb_clients'])
nb_rounds = int(client_cfg['nb_rounds'])
device = client_cfg['device']
nb_device = int(client_cfg['nb_device'])
params = model.get_weights()
print('Reading {} as the configuration file'.format(config_file))
print('Creating a {} model'.format(model_name))


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


strategy = fl.server.strategy.FedAvg(
    initial_parameters=fl.common.ndarrays_to_parameters(params),
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
)


client_resources = None
DEVICE = torch.device(device)
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": nb_device}


def simulation():
    results = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=nb_clients,
        config=fl.server.ServerConfig(num_rounds=nb_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    return results
