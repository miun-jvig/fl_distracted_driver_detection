from fl.simulation import simulation
from utils.evaluation import evaluation
from fl.client import training_history
from fl.clientdata import load_test_data_and_model
from config.configloader import model_cfg
import os

# data
models_dir = model_cfg['models_dir']
model_name = model_cfg['model_name']


def main():
    # start simulation
    simulation()
    test = training_history
    print('training_history is of size {}'.format(len(test)))
    # evaluate results
    model, xt, yt = load_test_data_and_model()
    print('Evaluating the model on the test set and store everything in {}'.format(models_dir))
    evaluation(model, xt, yt, training_history, os.path.join(models_dir, model_name))


if __name__ == '__main__':
    main()
