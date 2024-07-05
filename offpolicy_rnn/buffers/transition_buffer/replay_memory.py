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
    def __init__(self, max_transition_num: int=1000000, max_traj_step: Optional[int]=1000, rnn_slice_length=1):
        self.max_transition_num = max_transition_num
        self.memory: List[Transition] = []
        self.trajectory_length = []
        self.trajectory_start = []
        self.memory_buffer: Optional[np.ndarray] = None
        self.ind_range: Optional[List[List[int]]] = None
        self.name2range: Optional[Dict[str, List[int]]] = dict()
        self.ptr = 0
        self.max_traj_step = max_traj_step
        self.transition_buffer: List[int] = []
        self.transition_count = 0
        self.rnn_slice_length = rnn_slice_length
        self._last_saving_time = 0
        self._last_saving_size = 0

        self._last_sampled_batch_np = None

    @property
    def available_traj_num(self):
        return len(self.trajectory_length)

    def reset(self):
        self.memory = []
        self.trajectory_length = []
        self.trajectory_start = []
        self.ptr = 0
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
        traj_len = [self.trajectory_length[ind] for ind in traj_inds]
        max_traj_len = max(traj_len)
        max_traj_len = self.get_max_len(max_traj_len, self.rnn_slice_length)
        total_size = sum(traj_len)
        # merge trajs
        if self._last_sampled_batch_np is None:
            self._last_sampled_batch_np = np.zeros((len(traj_len), self.max_traj_step, self.memory_buffer.shape[-1]))
        else:
            if self._last_sampled_batch_np.shape[0] < len(traj_len):
                self._last_sampled_batch_np = np.zeros((len(traj_len), self.max_traj_step, self.memory_buffer.shape[-1]))
            elif self._last_sampled_batch_np.shape[0] > len(traj_len) * 2:
                self._last_sampled_batch_np = np.zeros((len(traj_len), self.max_traj_step, self.memory_buffer.shape[-1]))
        self._last_sampled_batch_np[:len(traj_len), :] = 0
        for idx, traj_ind in enumerate(traj_inds):
            self._last_sampled_batch_np[idx, :traj_len[idx], :] = self.memory_buffer[self.trajectory_start[traj_ind]:self.trajectory_start[traj_ind] + traj_len[idx], :]
        trajs = self._last_sampled_batch_np[:len(traj_len), :max_traj_len].copy()
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
        self.memory_buffer = np.zeros((int(self.max_transition_num + self.max_traj_step), dim))
        print(f'buffer init done!')

    def _init_memory_buffer(self, transition: Transition):
        print('init replay buffer')
        start_dim = 0
        self.ind_range = []
        self.trajectory_length = []
        self.trajectory_start = []
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

    def _insert_transition(self, transition: Transition):
        self.memory_buffer[self.ptr, :] = self.transition_to_array(transition)
        self.transition_buffer.append(self.ptr)

    def complete_traj(self, memory: List[Transition]):
        if self.memory_buffer is None:
            self._init_memory_buffer(memory[0])
        traj_len = len(memory)
        # clear trajectory_start, trajectory_length, transition_count, transition_buffer
        remove_traj_num = 0
        if self.transition_count + traj_len > self.max_transition_num:
            transition_count = self.transition_count
            while transition_count + traj_len > self.max_transition_num:
                transition_count -= self.trajectory_length[remove_traj_num]
                remove_traj_num += 1

        if remove_traj_num > 0:
            removed_num = sum(self.trajectory_length[:remove_traj_num])
            self.transition_buffer[:removed_num] = []
            self.transition_count -= removed_num
            self.trajectory_start[:remove_traj_num] = []
            self.trajectory_length[:remove_traj_num] = []

        self.trajectory_start.append(self.ptr)
        for ind, transition in enumerate(memory):
            self.memory_buffer[self.ptr] = 0
            self._insert_transition(transition)
            self.ptr += 1
        self.trajectory_length.append(traj_len)
        self.transition_count += len(memory)
        if self.ptr >= self.max_transition_num:
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


    def __len__(self) -> int:
        return len(self.trajectory_length)

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
        trajs = [self.memory_buffer[traj_ind] for traj_ind
                 in res]
        trajs = np.array(trajs, copy=True)
        res = self.array_to_transition(trajs)
        return res


