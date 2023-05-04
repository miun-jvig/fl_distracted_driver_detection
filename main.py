from fl.server import simulation
from utils.evaluation import evaluation
from fl.clientdata import load_test_data
from models.model import load_model, create_lite_model, summary
from config.configloader import model_cfg, client_cfg
import os

# data
models_dir = model_cfg['models_dir']
model_name = model_cfg['model_name']
nb_fl_rounds = client_cfg['nb_rounds']
use_lite_model = client_cfg['lite_model']


def main():
    # start simulation
    #simulation()

    # load model and test data
    xt, yt = load_test_data()
    model = load_model()
    filepath = "./logs/" + model_name + f"/server/cpft-{nb_fl_rounds}.ckpt"
    model.load_weights(filepath)

    # compress model to lite
    if use_lite_model is not None:
        model = create_lite_model(model, use_lite_model)

    # model.summary()
    # summary(int_model)

    # evaluate results
    print('Evaluating the model on the test set and store everything in {}'.format(models_dir))
    evaluation(model, model_name, xt, yt, os.path.join(models_dir, model_name), use_lite_model)


if __name__ == '__main__':
    main()
