from fl.client import client_fn
import flwr as fl


############### TEST AREA ###############
# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=1),
)
############### TEST AREA ###############
