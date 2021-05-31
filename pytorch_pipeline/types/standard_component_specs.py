TRAINER_MODULE_FILE = "module_file"
TRAINER_DATA_MODULE_FILE = "data_module_file"
TRAINER_DATA_MODULE_ARGS = "data_module_args"
TRAINER_MODULE_ARGS = "module_file_args"
PTL_TRAINER_ARGS = "trainer_args"

TRAINER_MODEL_SAVE_PATH = "model_save_path"
PTL_TRAINER_OBJ = "ptl_trainer"


class Parameters:
    def __init__(self, type=None, optional=False):  # pylint: disable=redefined-builtin
        self.type = type
        self.optional = optional


class TrainerSpec:
    INPUT_DICT = {
        TRAINER_MODULE_FILE: Parameters(type=str),
        TRAINER_DATA_MODULE_FILE: Parameters(type=str),
    }

    OUTPUT_DICT = {}

    EXECUTION_PROPERTIES = {
        TRAINER_DATA_MODULE_ARGS: Parameters(type=dict, optional=True),
        TRAINER_MODULE_ARGS: Parameters(type=dict),
        PTL_TRAINER_ARGS: Parameters(type=dict, optional=True),
    }
