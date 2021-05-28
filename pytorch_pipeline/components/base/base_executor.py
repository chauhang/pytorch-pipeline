import abc
from six import with_metaclass


class BaseExecutor(with_metaclass(abc.ABCMeta, object)):
    def __init__(self):
        pass

    @abc.abstractmethod
    def Do(self, input_dict: dict, output_dict: dict, exec_properties: dict):
        pass
