from fl.client import client_fn
from config.configloader import client_cfg
import flwr as fl

# data from config
nb_clients = int(client_cfg['nb_clients'])
nb_rounds = int(client_cfg['nb_rounds'])


def main():
    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=nb_clients,
        config=fl.server.ServerConfig(num_rounds=nb_rounds),
    )


if __name__ == '__main__':
    main()
