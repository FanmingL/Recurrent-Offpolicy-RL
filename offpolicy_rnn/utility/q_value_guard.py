import torch


class QValueGuard:
    """
    avoid Q loss from diverging
    usage:
    target_Q = r + gamma * q_value_guard.clamp(next_q)
    q_value_guard.update(target_Q)
    """
    def __init__(self, guard_min=True, guard_max=True, decay_ratio=1.0):
        self._min = 1000000 if guard_min else None
        self._max = -1000000 if guard_max else None
        self._init_flag = True
        self._decay_ratio = decay_ratio

    def reset(self):
        self._min = 1000000 if self._min is not None else None
        self._max = -1000000 if self._max is not None else None
        self._init_flag = True

    def clamp(self, value: torch.Tensor):
        if self._init_flag:
            self._min = value.min().item() if self._min is not None else None
            self._max = value.max().item() if self._max is not None else None
            self._init_flag = False
        return value.clamp(min=self._min, max=self._max)

    def update(self, value):
        value_min = value.min().item()
        value_max = value.max().item()
        self._min = min(self._min, value_min if self._min is not None else None)
        self._max = max(self._max, value_max if self._max is not None else None)
        if self._decay_ratio < 1:
            if self._min is not None:
                self._min = self._decay_ratio * self._min + (1-self._decay_ratio) * value_min
            if self._max is not None:
                self._max = self._decay_ratio * self._max + (1-self._decay_ratio) * value_max

    def get_min(self) -> float:
        return self._min

    def get_max(self) -> float:
        return self._max

