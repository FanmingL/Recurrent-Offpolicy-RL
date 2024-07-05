import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.*")
warnings.filterwarnings("ignore", ".*The DISPLAY environment variable is missing") # dmc
warnings.filterwarnings("ignore", category=FutureWarning) # dmc
import multiprocessing
from offpolicy_rnn import init_smart_logger, Parameter, alg_init


def main():
    if not multiprocessing.get_start_method(allow_none=True) == 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    init_smart_logger()
    parameter = Parameter()
    sac = alg_init(parameter)
    sac.train()


if __name__ == '__main__':
    main()