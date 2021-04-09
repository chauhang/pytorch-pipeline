import pytorch_lightning as pl
from pytorch_pipeline.components.trainer.generic_executor import GenericExecutor

class Executor(GenericExecutor):

    def __init__(self):
        super(GenericExecutor, self).__init__()

    def Do(self, model_class, data_module_class=None, data_module_args=None, module_file_args=None):

        if data_module_class:
            dm = data_module_class(**data_module_args if data_module_args else {})
            dm.prepare_data()
            dm.setup(stage="fit")

            model = model_class(**module_file_args if module_file_args else {})
            max_epochs = module_file_args["max_epochs"]

            trainer = pl.Trainer(
                max_epochs=max_epochs,
            )

            trainer.fit(model, dm)
            trainer.test()
