from pytorch_pipeline.components.base.base_executor import BaseExecutor


class GenericExecutor(BaseExecutor):
    def Do(self, input_dict: dict, output_dict: dict, exec_properties: dict):
        # TODO: Code to train pretrained model
        pass

    def _GetFnArgs(self):
        pass
