import flwr as fl


def start_server(server_address, num_rounds):
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=num_rounds), server_address=server_address)
