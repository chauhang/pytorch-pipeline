import abc
from six import with_metaclass
import logging


class BaseExecutor(with_metaclass(abc.ABCMeta, object)):
    def __init__(self):
        pass

    @abc.abstractmethod
    def Do(self, input_dict: dict, output_dict: dict, exec_properties: dict):
        pass

    def _log_startup(self, input_dict: dict, output_dict: dict, exec_properties):
        """Log inputs, outputs, and executor properties in a standard format."""
        class_name = self.__class__.__name__
        logging.debug("Starting {} execution.".format(class_name))
        logging.debug("Inputs for {} are: {}".format(class_name, input_dict))
        logging.debug("Outputs for {} are: {}".format(class_name, output_dict))
        logging.debug("Execution Properties for {} are: {}".format(class_name, exec_properties))
