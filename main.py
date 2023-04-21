from fl.simulation import simulation
from utils.evaluation import evaluation
from fl.clientdata import load_test_data, load_model
from config.configloader import model_cfg, client_cfg
import os

# data
models_dir = model_cfg['models_dir']
model_name = model_cfg['model_name']
nb_fl_rounds = client_cfg['nb_rounds']


def main():
    # start simulation
    simulation()

    # load model and test data
    xt, yt = load_test_data()
    model = load_model()
    filepath = "./logs/" + model_name + f"/server/cpft-{nb_fl_rounds}.ckpt"
    model.load_weights(filepath)

    # evaluate results
    print('Evaluating the model on the test set and store everything in {}'.format(models_dir))
    evaluation(model, model_name, xt, yt, os.path.join(models_dir, model_name))


if __name__ == '__main__':
    main()
