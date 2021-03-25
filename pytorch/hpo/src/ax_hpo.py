import os
import sys
import argparse
from ax.service.ax_client import AxClient
import pytorch_lightning as pl
import importlib
import json

def train_evaluate(params, max_epochs=100):
    model = Model(**params)
    dm = DataModule()
    dm.prepare_data()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    test_accuracy = trainer.callback_metrics.get("test_acc")
    return test_accuracy

def model_training_hyperparameter_tuning(max_epochs, total_trials, params):
    """
     This function takes input params max_epochs, total_trials, params
     and creates a nested run in Mlflow. The parameters, metrics, model and summary are dumped into their
     respective mlflow-run ids. The best parameters are dumped along with the baseline model.

    :param max_epochs: Max epochs used for training the model. Type:int
    :param total_trials: Number of ax-client experimental trials. Type:int
    :param params: Model parameters. Type:dict
    """
    # train_evaluate(params=params, max_epochs=max_epochs)

    ax_client = AxClient()
    ax_client.create_experiment(
        # parameters=[
        #     {"name": "lr", "type": "range", "bounds": [1e-3, 0.15], "log_scale": True},
        #     {"name": "weight_decay", "type": "range", "bounds": [1e-4, 1e-3]},
        #     {"name": "momentum", "type": "range", "bounds": [0.7, 1.0]},
        # ],
        parameters=params,
        objective_name="test_accuracy"
    )

    for i in range(total_trials):
        parameters, trial_index = ax_client.get_next_trial()
        test_accuracy = train_evaluate(params=parameters, max_epochs=max_epochs)

        # completion of trial
        ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())

    best_parameters, metrics = ax_client.get_best_parameters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    parser.add_argument(
        "--total_trials",
        default=3,
        help="umber of trials to be run for the optimization experiment",
    )

    parser.add_argument(
        "--model_file",
        help="model file",
    )

    parser.add_argument(
        "--data_module_file",
        help="data module file",
    )

    parser.add_argument(
        "--params_file",
        help="ax parameters",
    )

    args = parser.parse_args()

    if "max_epochs" in args:
        max_epochs = args.max_epochs
    else:
        max_epochs = 100

    input_model_file = args.model_file
    input_data_module_file = args.data_module_file

    params_file = args.params_file

    with open(params_file) as f:
        data = f.read()
          
    params = json.loads(data)

    sys.path.append(os.path.abspath(os.path.dirname(input_model_file)))

    from model_file import Model
    from data_module_file import DataModule
    
    model_training_hyperparameter_tuning(
        max_epochs=int(max_epochs), total_trials=int(args.total_trials), params=params
    )
