from fl.client import client_fn
from fl.clientdata import model
from config.configloader import client_cfg, config_file, model_cfg
import flwr as fl

# data from config
model_name = model_cfg['model_name']
nb_clients = int(client_cfg['nb_clients'])
nb_rounds = int(client_cfg['nb_rounds'])
params = model.get_weights()
print('Reading {} as the configuration file'.format(config_file))
print('Creating a {} model'.format(model_name))


def main():
    strategy = fl.server.strategy.FedAvg(initial_parameters=fl.common.ndarrays_to_parameters(params))
    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=nb_clients,
        config=fl.server.ServerConfig(num_rounds=nb_rounds),
        strategy=strategy,
    )


if __name__ == '__main__':
    main()
