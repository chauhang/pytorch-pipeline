import pytorch_lightning as pl
import torch
import os
from typing import Optional
from pathlib import Path
from pytorch_pipeline.components.trainer.generic_executor import GenericExecutor

# from pytorch_pipeline.components.utils.lib_minio import LibMinio


class Executor(GenericExecutor):
    def __init__(self):
        super(GenericExecutor, self).__init__()

    def Do(
        self,
        model_class,
        data_module_class=None,
        data_module_args: Optional[dict] = None,
        module_file_args: Optional[dict] = None,
        trainer_args: Optional[dict] = None,
    ):

        if data_module_class:
            dm = data_module_class(**data_module_args if data_module_args else {})
            dm.prepare_data()
            dm.setup(stage="fit")

            # parser = module_file_args
            # args = vars(parser.parse_args())
            model = model_class(**module_file_args if module_file_args else {})

            from argparse import Namespace

            trainer_args.update(module_file_args)
            print("/n/n This is trainer args",trainer_args)
            parser = Namespace(**trainer_args)
            trainer = pl.Trainer.from_argparse_args(parser)

            trainer.fit(model, dm)
            trainer.test()
            test_accuracy = trainer.callback_metrics.get("avg_test_acc")

            if "checkpoint_dir" in module_file_args:
                model_save_path = module_file_args["checkpoint_dir"]
            else:
                model_save_path = "/tmp"

            if "model_name" in module_file_args:
                model_name = module_file_args["model_name"]
            else:
                model_name = "model_state_dict.pth"

            Path(model_save_path).mkdir(parents=True, exist_ok=True)
            model_save_path = os.path.join(model_save_path, model_name)

            torch.save(model.state_dict(), model_save_path)
            print("Saving model to {}".format(model_save_path))

            return trainer
