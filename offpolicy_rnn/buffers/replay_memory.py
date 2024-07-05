import copy
from collections import namedtuple
import numpy as np
import pickle
import time
from typing import List, Union, Tuple, Dict, Optional, Any


# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
tuplenames = ('state', 'last_state', 'last_action', 'action', 'next_state', 'reward', 'logp', 'mask', 'start', 'done', 'reward_input', 'timeout')
Transition = namedtuple('Transition', tuplenames)


class MemoryArray(object):
    def __init__(self, max_trajectory_num: int=1000, max_traj_step: int=1000, rnn_slice_length: int=1):
        self.memory: List[Transition] = []
        self.trajectory_length = [0] * max_trajectory_num
        self.max_trajectory_num = max_trajectory_num
        self.available_traj_num = 0
        self.memory_buffer: Optional[np.ndarray] = None
        self.ind_range: Optional[List[List[int]]] = None
        self.name2range: Optional[Dict[str, List[int]]] = dict()
        self.ptr = 0
        self.max_traj_step = max_traj_step
        self.transition_buffer: List[Tuple[int, int]] = []
        self.transition_count = 0
        self.rnn_slice_length = rnn_slice_length
        self._last_saving_time = 0
        self._last_saving_size = 0

    def reset(self):
        self.memory = []
        self.trajectory_length = [0] * self.max_trajectory_num
        self.available_traj_num = 0
        self.ptr = 0
        self.max_traj_step = self.max_traj_step
        self.transition_buffer = []
        self.transition_count = 0
        self._last_saving_time = 0
        self._last_saving_size = 0

    @staticmethod
    def get_max_len(max_len: int, slice_length: int):
        if max_len % slice_length == 0 and max_len > 0:
            return max_len
        else:
            max_len = (max_len // slice_length + 1) * slice_length
        return max_len

    def _traj_ind_sample(self, batch_size, max_sample_size) -> np.ndarray:
        mean_traj_len = self.transition_count / self.available_traj_num
        if batch_size is not None:
            desired_traj_num = int(np.ceil(batch_size / mean_traj_len))
        else:
            desired_traj_num = self.available_traj_num
        if max_sample_size is not None:
            max_traj_num = int(np.ceil(max_sample_size / self.max_traj_step))
            desired_traj_num = min(desired_traj_num, max_traj_num)
        rnd_permutation = np.random.permutation(self.available_traj_num)
        if batch_size is None:
            traj_inds = np.arange(self.available_traj_num)
        else:
            if desired_traj_num <= self.available_traj_num:
                traj_inds = rnd_permutation[:int(desired_traj_num)]
            else:
                traj_inds = np.random.randint(0, self.available_traj_num, (int(desired_traj_num),))
        traj_len = [self.trajectory_length[ind] for ind in traj_inds]
        sample_sum = sum(traj_len)
        traj_num = len(traj_len)
        additional_traj_inds = []
        while sample_sum < batch_size and (max_sample_size is None or traj_num < max_traj_num):
            traj_num += 1
            target_ind = desired_traj_num + len(additional_traj_inds)
            if self.available_traj_num > target_ind:
                idx = rnd_permutation[target_ind]
                sample_sum += self.trajectory_length[idx]
                additional_traj_inds.append(idx)
            else:
                target_ind = np.random.randint(low=0, high=self.available_traj_num)
                sample_sum += self.trajectory_length[target_ind]
                additional_traj_inds.append(target_ind)
        if len(additional_traj_inds):
            traj_inds = np.concatenate((traj_inds, np.array(additional_traj_inds)), axis=0)
        return traj_inds

    def sample_trajs(self, batch_size, max_sample_size=None, get_all=False) -> Tuple[Transition, int]:
        if get_all:
            traj_inds = np.arange(self.available_traj_num)
        else:
            traj_inds = self._traj_ind_sample(batch_size, max_sample_size)
        # this step is cost!
        trajs = self.memory_buffer[traj_inds]
        traj_len = [self.trajectory_length[ind] for ind in traj_inds]
        max_traj_len = max(traj_len)
        max_traj_len = self.get_max_len(max_traj_len, self.rnn_slice_length)
        trajs = trajs[:, :max_traj_len, :]
        total_size = sum(traj_len)

        return self.array_to_transition(trajs), total_size

    def get_all_trajs(self) -> Tuple[Transition, int]:
        return self.sample_trajs(None, None, get_all=True)

    def transition_to_array(self, transition: Transition):
        res = []

        for item in transition:
            if isinstance(item, np.ndarray):
                res.append(item.reshape((1, -1)))
            elif isinstance(item, list):
                res.append(np.array(item).reshape((1, -1)))
            elif item is None:
                pass
            elif np.isscalar(item):
                res.append(np.array([[item]]))
            else:
                raise NotImplementedError('not implement for type of {}'.format(type(item)))
        res = np.hstack(res)
        assert res.shape[-1] == self.memory_buffer.shape[-1], 'data_size: {}, buffer_size: {}'.format(res.shape, self.memory_buffer.shape)
        return res

    def array_to_transition(self, data: np.ndarray) -> Transition:
        data_list = []
        for item in self.ind_range:
            if len(item) > 0:
                start = item[0]
                end = item[-1] + 1
                data_list.append(data[..., start:end])
            else:
                data_list.append(None)
        res = Transition(*data_list)
        return res


    def _make_buffer(self, dim):
        self.memory_buffer = np.zeros((self.max_trajectory_num, self.max_traj_step, dim))
        print(f'buffer init done!')

    def _init_memory_buffer(self, transition: Transition):
        print('init replay buffer')
        start_dim = 0
        self.ind_range = []
        self.trajectory_length = [0] * self.max_trajectory_num
        end_dim = 0
        for item in transition:
            dim = 0
            if isinstance(item, np.ndarray):
                dim = item.shape[-1]
            elif isinstance(item, list):
                dim = len(item)
            elif item is None:
                dim = 0
            elif np.isscalar(item):
                dim = 1
            end_dim = start_dim + dim
            self.ind_range.append(list(range(start_dim, end_dim)))
            start_dim = end_dim
        for name, ind_range in zip(tuplenames, self.ind_range):
            self.name2range[name] = ind_range
            print(f'name: {name}, ind: {ind_range}')
        self._make_buffer(end_dim)

    def _insert_transition(self, transition: Transition, ind: int):
        self.memory_buffer[self.ptr, ind, :] = self.transition_to_array(transition)
        self.transition_buffer.append((self.ptr, ind))

    def complete_traj(self, memory: List[Transition]):
        if self.memory_buffer is None:
            self._init_memory_buffer(memory[0])
        self.memory_buffer[self.ptr] = 0
        for ind, transition in enumerate(memory):
            self._insert_transition(transition, ind)
        self.transition_count -= self.trajectory_length[self.ptr]
        if self.trajectory_length[self.ptr] > 0:
            self.transition_buffer[:self.trajectory_length[self.ptr]] = []
        self.trajectory_length[self.ptr] = len(memory)

        self.ptr += 1
        self.available_traj_num = max(self.available_traj_num, self.ptr)
        self.transition_count += len(memory)
        if self.ptr >= self.max_trajectory_num:
            self.ptr = 0


    def mem_push(self, transition: Transition, parallel_num=1, valid_data=True):
        if not valid_data:
            self.memory = []
            return
        self.memory.append(transition)
        done = transition.done if np.isscalar(transition.done) else np.array(transition.done)
        mask = transition.mask if np.isscalar(transition.mask) else np.array(transition.mask)
        if np.all(done):
            if np.all(mask):
                if parallel_num == 1:
                    self.complete_traj(self.memory)
                else:
                    memory_list = []
                    for i in range(parallel_num):
                        memory = []
                        for item in self.memory:
                            _transition = Transition(*[item2[i] if (item2 is not None) and (not np.isscalar(item2)) else item2 for item2 in item])
                            memory.append(_transition)
                        memory_list.append(memory)
                    for memory in memory_list:
                        self.complete_traj(memory)
            self.memory = []


    def traj_push(self, trajectories: Transition):
        assert len(trajectories.mask.shape) == 3
        traj_num = trajectories.mask.shape[0]
        traj_len = trajectories.mask.shape[1]
        for traj_id in range(traj_num):
            for step in range(traj_len):
                if trajectories.mask[traj_id, step, 0] == 0:
                    break
                else:
                    transition = Transition(*[item[traj_id, step, :] if item is not None else None for item in trajectories])
                    self.mem_push(transition)

    def mem_list_push(self, transition_list: List[Transition]):
        for item in transition_list:
            self.mem_push(item)

    def __len__(self) -> int:
        return self.available_traj_num

    @property
    def size(self) -> int:
        return self.transition_count

    def save_to_disk(self, path):
        self._last_saving_time = time.time()
        self._last_saving_size = self.size
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load_from_disk(path) -> "MemoryArray":
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def sample_transitions(self, batch_size: Optional[int]=None) -> Transition:
        if batch_size is not None:
            list_ind = np.random.randint(0, self.transition_count, (batch_size,))
        else:
            list_ind = list(range(self.transition_count))
        res = [self.transition_buffer[ind] for ind in list_ind]
        trajs = [self.memory_buffer[traj_ind, point_ind] for traj_ind, point_ind
                 in res]
        trajs = np.array(trajs, copy=True)
        res = self.array_to_transition(trajs)
        return res


