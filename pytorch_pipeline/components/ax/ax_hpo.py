from ax.service.ax_client import AxClient
from pytorch_pipeline.components.trainer.component import Trainer
from pytorch_pipeline.components.minio.component import MinIO
from ax.plot.contour import interact_contour, plot_contour
import os
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse


class AxOptimization:
    def __init__(self, total_trials, params):
        self.total_trials = total_trials
        self.params = params

    def run_ax_get_best_parameters(
        self,
        module_file_args,
        data_module_args,
        trainer_args,
        model_file_name,
        data_module_file_name,
    ):

        ax_client = AxClient()
        ax_client.create_experiment(parameters=self.params, objective_name="test_accuracy")
        ax_param_trial_dict = []
        model_name_list = []

        for i in range(self.total_trials):
            parameters, trial_index = ax_client.get_next_trial()
            print(trial_index)
            print(i)
            ax_param_trial_dict.append(parameters)
            current_model_name = "model.pth"
            if "model_name" in module_file_args:
                current_model_name = module_file_args["model_name"]

            model_name = (
                current_model_name.split(".")[0]
                + str(trial_index)
                + "."
                + current_model_name.split(".")[1]
            )

            model_name_list.append(model_name)

            module_file_args["model_name"] = model_name

            print("Name of Model", model_name)
            trainer = Trainer(
                module_file=model_file_name,
                data_module_file=data_module_file_name,
                module_file_args=module_file_args,
                data_module_args=data_module_args,
                trainer_args=trainer_args,
            )
            test_accuracy = trainer.ptl_trainer.callback_metrics.get("avg_test_acc")
            print(test_accuracy)
            ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())
            module_file_args["model_name"] = current_model_name

        best_parameters, metrics = ax_client.get_best_parameters()
        trials_df = ax_client.get_trials_data_frame()
        trials_df.to_csv("summary.csv")
        columns = trials_df.columns
        self.columns = columns
        print("This is sum url", module_file_args["summary_url"])
        parse_obj = urlparse(module_file_args["summary_url"], allow_fragments=False)
        bucket_name = parse_obj.netloc
        print(bucket_name)
        folder_name = str(parse_obj.path).lstrip("/")

        # TODO:
        endpoint = "minio-service.kubeflow:9000"
        MinIO(
            source="summary.csv",
            bucket_name=bucket_name,
            destination=folder_name,
            endpoint=endpoint,
        )
        best_trail_index = ax_param_trial_dict.index(best_parameters)
        model_name_for_mar = model_name_list[best_trail_index]

        from ax.utils.notebook.plotting import render, init_notebook_plotting

        # model = trainer.pl_trainer.model
        # render(plot_contour(model=model, param_x="x1", param_y="x2", metric_name='hartmann6'))
        self.model_name_for_mar = model_name_for_mar
        print(model_name_for_mar)
        shutil.move(
            os.path.join(module_file_args["checkpoint_dir"], model_name_for_mar),
            os.path.join(module_file_args["checkpoint_dir"], "mnist_best.pth"),
        )
        print("Best Trial Index", best_trail_index)
        print(best_parameters)

        save_path = module_file_args["save_path"]

        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        # best_parameters_file = os.path.join(os.path.abspath(save_path), 'best_parameters.json')

        with open(save_path, "w") as fp:
            json.dump(best_parameters, fp)
