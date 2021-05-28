import abc
from pytorch_pipeline.types import standard_component_specs
from six import with_metaclass


class BaseComponent(with_metaclass(abc.ABCMeta, object)):
    def __init__(self):
        pass

    @classmethod
    def _validate_spec(
        cls,
        spec: standard_component_specs,
        input_dict: dict,
        output_dict: dict,
        exec_properties: dict,
    ):

        for key, value in input_dict.items():
            cls._type_check(actual_value=value, key=key, spec_dict=spec.INPUT_DICT)

        for key, value in output_dict.items():
            cls._type_check(actual_value=value, key=key, spec_dict=spec.OUTPUT_DICT)

        for key, value in exec_properties.items():
            cls._type_check(actual_value=value, key=key, spec_dict=spec.EXECUTION_PROPERTIES)

    @classmethod
    def _optional_check(cls, actual_value: any, key: str, spec_dict: dict):
        is_optional = spec_dict[key].optional

        if not is_optional and not actual_value:
            raise ValueError(
                "{key} is not optional. Received value: {actual_value}".format(
                    key=key, actual_value=actual_value
                )
            )

        return is_optional

    @classmethod
    def _type_check(cls, actual_value, key, spec_dict):
        if not actual_value:
            is_optional = cls._optional_check(
                actual_value=actual_value, key=key, spec_dict=spec_dict
            )
            if is_optional:
                return

        expected_type = spec_dict[key].type
        actual_type = type(actual_value)
        if actual_type != expected_type:
            raise TypeError(
                "{key} must be of type {expected_type} but received as {actual_type}".format(
                    key=key, expected_type=expected_type, actual_type=actual_type
                )
            )
