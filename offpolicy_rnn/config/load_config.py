import os
import smart_logger


def init_smart_logger():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.dirname(os.path.abspath(__file__))
    config_relative_path = os.path.relpath(config_path, base_path)
    smart_logger.init_config(os.path.join(config_relative_path, 'common_config.yaml'),
                             os.path.join(config_relative_path, 'experiment_config.yaml'),
                             base_path)
