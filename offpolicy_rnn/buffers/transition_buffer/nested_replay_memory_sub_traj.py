from .replay_memory import MemoryArray, Transition
import numpy as np
from typing import Tuple
import time
import math


class NestedMemoryFixedLengthArray(MemoryArray):
    def __init__(self, max_transition_num: int=1000, max_traj_step: int=1000,
                 rnn_slice_length: int=1, additional_history_len: int=0,
                 map_to_two_power=True, fixed_traj_len=-1):
        _max_traj_len = max_traj_step+1+additional_history_len
        if map_to_two_power:
            _max_traj_len = self.nearest_power_of_two(_max_traj_len)
            if _max_traj_len >= 2048:
                print(f'[ WARNING ] current maximum traj len is set to {_max_traj_len}, which could slow down '
                      f'the program speed. You\'d better set `map_to_two_power` to False in this case.')
        super().__init__(max_transition_num, _max_traj_len , rnn_slice_length)
        self._last_grouped_traj_np = None
        self._source_range = None
        self._target_range = None
        self._mask_range = None
        self._additional_history_len = additional_history_len
        self._skip_step = 1 + self._additional_history_len
        self._fixed_traj_len = fixed_traj_len
        if self._fixed_traj_len <= 0:
            self._fixed_traj_len = 111111111111

    @staticmethod
    def nearest_power_of_two(x):
        # 计算0.5 * x
        target = x
        # 计算对数，并四舍五入到最接近的整数
        nearest_exp = int(math.ceil(math.log(target, 2)))
        # 计算2的nearest_exp次幂
        if nearest_exp < 0:
            nearest_exp = 0
        nearest_power = int(math.ceil(2 ** nearest_exp))
        return nearest_power


    def load_equalize(self, traj_lens, max_traj_length):
        traj_sorted_idx = np.arange(len(traj_lens))
        # traj_sorted_idx = traj_lens
        bins = []
        bin_cap = []
        for idx in traj_sorted_idx:
            traj_len = traj_lens[idx]
            if len(bins) > 0:
                bin_reserves = [item - traj_len if item > traj_len else max_traj_length + 1 for item in bin_cap]
                optimal_bin_idx = np.argmin(bin_reserves)
                if bin_reserves[optimal_bin_idx] <= max_traj_length:
                    bins[optimal_bin_idx].append(idx)
                    bin_cap[optimal_bin_idx] = bin_reserves[optimal_bin_idx]
                    continue
            bins.append([])
            bin_cap.append(max_traj_length)
            bins[-1].append(idx)
            bin_cap[-1] -= traj_len
        return bins

    def _init_memory_buffer(self, transition: Transition):
        super()._init_memory_buffer(transition)
        # target
        next_state_range = self.name2range['next_state']
        reward_range = self.name2range['reward']
        # action_range = self.name2range['action']

        # source
        state_range = self.name2range['state']
        last_state_range = self.name2range['last_state']
        reward_input_range = self.name2range['reward_input']
        # last_action_range = self.name2range['last_action']
        source_range = state_range + reward_input_range + last_state_range
        target_range = next_state_range + reward_range + state_range
        self._source_range = source_range
        self._target_range = target_range
        self._action_range = self.name2range['action']
        self._mask_range = self.name2range['mask']
        self._rnn_start_range = self.name2range['start']

    def _mask_rnd_select(self, mask, select_num):
        mask_1d = mask.reshape((-1,))
        mask_idx = mask_1d.nonzero()[0]
        mask_idx_set_zero = mask_idx[np.random.permutation(mask_idx.shape[0])[:-select_num]]
        mask_1d[mask_idx_set_zero] = 0

    def get_equalized_valid_num_each_traj(self, traj_len_added_1, desired_total_valid_number):
        # traj_len_added_1 = [item + 1 for item in traj_len]
        argsort_idx = np.argsort(traj_len_added_1)
        valid_num = 0
        total_traj_num = len(traj_len_added_1)
        average_num = int(np.ceil(desired_total_valid_number / total_traj_num))
        valid_num_list = [average_num for _ in range(total_traj_num)]
        for i in range(total_traj_num):
            traj_len = traj_len_added_1[argsort_idx[i]] - 1
            desired_num = int(np.ceil((desired_total_valid_number - valid_num) / (total_traj_num - i)))
            if desired_num <= 0:
                desired_num = average_num
            if desired_num > traj_len:
                desired_num = traj_len
            valid_num += desired_num
            valid_num_list[argsort_idx[i]] = desired_num
        return valid_num_list


    def _traj_ind_sample(self, batch_size, max_sample_size) -> np.ndarray:
        mean_traj_len = self._fixed_traj_len
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
        traj_len = [min(self.trajectory_length[ind], self._fixed_traj_len) for ind in traj_inds]
        sample_sum = sum(traj_len)
        traj_num = len(traj_len)
        additional_traj_inds = []
        while sample_sum < batch_size and (max_sample_size is None or traj_num < max_traj_num):
            traj_num += 1
            target_ind = desired_traj_num + len(additional_traj_inds)
            if self.available_traj_num > target_ind:
                idx = rnd_permutation[target_ind]
                sample_sum += min(self.trajectory_length[idx], self._fixed_traj_len)
                additional_traj_inds.append(idx)
            else:
                target_ind = np.random.randint(low=0, high=self.available_traj_num)
                sample_sum += min(self.trajectory_length[target_ind], self._fixed_traj_len)
                additional_traj_inds.append(target_ind)
        if len(additional_traj_inds):
            traj_inds = np.concatenate((traj_inds, np.array(additional_traj_inds)), axis=0)
        return traj_inds

    def sample_trajs(self, batch_size, max_sample_size=None, get_all=False,
                     randomize_mask=False, valid_number_post_randomized=0,
                     equalize_data_of_each_traj=False, random_trunc_traj=False,
                     copy=False, nest_stack_trajs=True, clip_size=0) -> Tuple[Transition, int, np.ndarray]:
        if get_all:
            traj_inds = np.arange(self.available_traj_num)
        else:
            if random_trunc_traj:
                batch_size *= 2
            traj_inds = self._traj_ind_sample(batch_size, max_sample_size)
        # trajs = self.memory_buffer[traj_inds]
        # reserve 1 empty transition before each trajectory
        traj_len = [min(self._fixed_traj_len, self.trajectory_length[ind]) + self._skip_step if not random_trunc_traj else np.random.randint(0, self.trajectory_length[ind]) + 1 + self._skip_step for ind in traj_inds]
        traj_start = [self.trajectory_start[ind] + np.random.randint(0, max(self.trajectory_length[ind] - self._fixed_traj_len, 1)) for ind in traj_inds]
        if randomize_mask and equalize_data_of_each_traj:
            valid_nums = self.get_equalized_valid_num_each_traj(traj_len, valid_number_post_randomized)
        # trajectory distribution
        max_traj_step = self.nearest_power_of_two(max(traj_len))
        # max_traj_step = self.max_traj_step
        if nest_stack_trajs:
            grouped_traj = self.load_equalize(traj_len, max_traj_step)
        else:
            grouped_traj = [[i] for i in range(len(traj_len))]
        nested_traj_num = len(grouped_traj)
        # max_traj_len = max(traj_len)
        # max_traj_len = self.get_max_len(max_traj_len, self.rnn_slice_length)
        # trajs = trajs[:, :max_traj_len, :]
        total_size = int(sum(traj_len) - len(traj_len) * self._skip_step)
        if self._last_grouped_traj_np is None:
            # make a new traj_np
            self._last_grouped_traj_np = np.zeros((nested_traj_num, self.max_traj_step, self.memory_buffer.shape[-1]))
            grouped_traj_np = self._last_grouped_traj_np
        else:
            if self._last_grouped_traj_np.shape[0] < nested_traj_num:
                # reinitialize a new traj_np
                self._last_grouped_traj_np = np.zeros((nested_traj_num, self.max_traj_step, self.memory_buffer.shape[-1]))
                grouped_traj_np = self._last_grouped_traj_np
            else:
                # reuse the last traj_np
                grouped_traj_np = self._last_grouped_traj_np
                # reset the former part to zero
                grouped_traj_np[:nested_traj_num] = 0
        traj_valid_indicator = grouped_traj_np[:, :, self._mask_range[0]:self._mask_range[0]+1].copy()
        for i in range(nested_traj_num):
            ptr = 0
            for j in range(len(grouped_traj[i])):
                traj_id_ = grouped_traj[i][j]
                traj_len_ = traj_len[traj_id_]
                traj_start_ = traj_start[traj_id_]
                # traj_np[i, 0] = 0, traj_np[i, 1] = data0, traj_np[i, 2] = data1, ...
                # note that real data length is (traj_len - 1)
                grouped_traj_np[i, ptr+self._skip_step:ptr+traj_len_, :] = self.memory_buffer[traj_start_:traj_start_+(traj_len_-self._skip_step), :].copy()
                # part of traj_np[i, 0] will be set to s0, a0, r0, such that next_state = cat(s0, next_state)
                grouped_traj_np[i, ptr+self._skip_step-1, self._target_range] = self.memory_buffer[traj_start_, self._source_range].copy()
                grouped_traj_np[i, ptr+self._skip_step-1, self._action_range] = 0
                grouped_traj_np[i, ptr:ptr+self._skip_step, self._rnn_start_range[0]] = 1
                traj_valid_indicator[i, ptr+self._skip_step:ptr+traj_len_, :] = self.memory_buffer[traj_start_:traj_start_+(traj_len_-self._skip_step), self._mask_range[0]:self._mask_range[0]+1].copy()
                if randomize_mask and equalize_data_of_each_traj:
                    zeros_idx = np.random.permutation(traj_len_-self._skip_step)[:-valid_nums[traj_id_]] + ptr + self._skip_step
                    grouped_traj_np[i, zeros_idx, self._mask_range[0]] = 0
                if clip_size > 0:
                    if traj_len_-self._skip_step > clip_size:
                        start_idx = np.random.randint(0, traj_len_-self._skip_step - clip_size)
                        if start_idx > 0:
                            grouped_traj_np[i, ptr+self._skip_step:ptr+start_idx + self._skip_step, self._mask_range[0]] = 0
                        if start_idx + self._skip_step + clip_size < traj_len_:
                            grouped_traj_np[i, ptr+self._skip_step + start_idx + clip_size:ptr+traj_len_, self._mask_range[0]] = 0
                ptr += traj_len_
            grouped_traj_np[i, ptr:, self._rnn_start_range[0]] = 1
        # print(f'set time period: {time.time() - start_time}, grouped_traj_np: {grouped_traj_np.shape}')
        # if copy:
        #     result = self.array_to_transition(grouped_traj_np[:nested_traj_num, self._additional_history_len:, :].copy())
        # else:
        #     result = self.array_to_transition(grouped_traj_np[:nested_traj_num, self._additional_history_len:, :])
        # traj_valid_indicator = traj_valid_indicator[:nested_traj_num, self._additional_history_len:]

        if copy:
            result = self.array_to_transition(grouped_traj_np[:nested_traj_num, :max_traj_step, :].copy())
        else:
            result = self.array_to_transition(grouped_traj_np[:nested_traj_num, :max_traj_step, :])
        traj_valid_indicator = traj_valid_indicator[:nested_traj_num, :max_traj_step, :]
        if randomize_mask and not equalize_data_of_each_traj:
            self._mask_rnd_select(result.mask, valid_number_post_randomized)
        print(result.state.shape, traj_valid_indicator.shape)
        return result, total_size, traj_valid_indicator

