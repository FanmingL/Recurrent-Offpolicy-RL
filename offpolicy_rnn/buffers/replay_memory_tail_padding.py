import copy
from collections import namedtuple
import numpy as np
import pickle
import time
from typing import List, Union, Tuple, Dict, Optional, Any
from .replay_memory import Transition, tuplenames, MemoryArray


class MemoryArrayTailZeroPadding(MemoryArray):
    def __init__(self, max_trajectory_num: int=1000, max_traj_step: int=1000, rnn_slice_length: int=1, fixed_length: int=32):
        super().__init__(max_trajectory_num, max_traj_step, rnn_slice_length)

        self.fixed_length = fixed_length
        assert self.fixed_length >= 1
        self.last_sampled_batch = None

    def reset(self):
        super().reset()
        self.last_sampled_batch = None

    def sample_fix_length_sub_trajs(self, batch_size, fix_length):
        list_ind = np.random.randint(0, self.transition_count, batch_size)
        res = [self.transition_buffer[ind] for ind in list_ind]
        if self.last_sampled_batch is None or not self.last_sampled_batch.shape[0] == batch_size or not self.last_sampled_batch.shape[1] == fix_length:
            trajs = [self.memory_buffer[traj_ind, point_ind:point_ind+self.fixed_length]for traj_ind, point_ind in res]
            trajs = np.array(trajs, copy=True)
            self.last_sampled_batch = trajs
        else:
            # reuse last sampled batch, avoid frequently allocating memory
            for ind, (traj_ind, point_ind) in enumerate(res):
                self.last_sampled_batch[ind, :, :] = self.memory_buffer[traj_ind,
                                                        point_ind: point_ind+self.fixed_length, :]

        res = self.array_to_transition(self.last_sampled_batch)
        return res

    def _make_buffer(self, dim):
        self.memory_buffer = np.zeros((self.max_trajectory_num, self.max_traj_step + self.fixed_length - 1, dim))
        print(f'Tail zero padding buffer init done!')


