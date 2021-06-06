from pytorch_kfp_components.components.base.base_component import BaseComponent
from pytorch_kfp_components.components.minio.executor import Executor
from pytorch_kfp_components.types import standard_component_specs


class MinIO(BaseComponent):
    def __init__(
        self,
        source: str,
        bucket_name: str,
        destination: str,
        endpoint: str,
    ):
        super(BaseComponent, self).__init__()

        input_dict = {
            standard_component_specs.MINIO_SOURCE: source,
            standard_component_specs.MINIO_BUCKET_NAME: bucket_name,
            standard_component_specs.MINIO_DESTINATION: destination,
        }

        output_dict = {}

        exec_properties = {
            standard_component_specs.MINIO_ENDPOINT: endpoint,
        }

        spec = standard_component_specs.MinIoSpec()
        self._validate_spec(
            spec=spec,
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
        )

        Executor().Do(
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
        )

        self.output_dict = output_dict
