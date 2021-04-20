import pytorch_lightning as pl
from pytorch_pipeline.components.trainer.generic_executor import GenericExecutor

class Executor(GenericExecutor):

    def __init__(self):
        super(GenericExecutor, self).__init__()

    def Do(self, model_class, data_module_class=None, data_module_args=None, module_file_args=None, trainer_args=None):

        if data_module_class:
            dm = data_module_class(**data_module_args if data_module_args else {})
            dm.prepare_data()
            dm.setup(stage="fit")

            parser = module_file_args
            args = vars(parser.parse_args())
            model = model_class(**args if args else {})

            trainer = pl.Trainer.from_argparse_args(parser, **trainer_args)

            trainer.fit(model, dm)
            trainer.test()
