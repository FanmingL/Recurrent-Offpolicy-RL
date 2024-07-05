import math

class CosineScheduler:
    def __init__(self, total_steps, initial_value, min_value):
        """
        初始化余弦学习率调度器
        :param total_steps: 总轮数
        :param initial_value: 初始学习率
        :param min_value: 最小学习率
        """
        self.total_steps = total_steps
        self.initial_value = initial_value
        self.min_value = min_value
        self.current_step = 0
        self._value = None
        self._last_value = None

    def validate(self):
        if self._value < self.min_value:
            self._value = self.min_value

        if self._last_value is not None:
            if self._last_value < self._value:
                self._value = self._last_value
        else:
            self._last_value = self._value

    def step(self):
        """
        更新到下一步，计算当前的学习率
        """
        self.current_step += 1

        self._value = self.min_value + 0.5 * (self.initial_value - self.min_value) * (1 + math.cos(self.current_step / self.total_steps * math.pi))
        self.validate()
        return self._value

    def get_value(self):
        """
        获取当前的学习率
        """
        if self.current_step == 0:
            return self.initial_value
        else:
            return self._value

class LinearScheduler:
    def __init__(self, total_steps, initial_value, end_value):
        """
        初始化余弦学习率调度器
        :param total_steps: 总轮数
        :param initial_value: 初始学习率
        :param min_value: 最小学习率
        """
        self.total_steps = total_steps
        self.initial_value = initial_value
        self.end_value = end_value
        self.current_step = 0
        self._value = self.initial_value

    def validate(self):
        if self.current_step >= self.total_steps:
            self._value = self.end_value

    def step(self):
        """
        更新到下一步，计算当前的学习率
        """
        self.current_step += 1
        self._value = self.current_step / self.total_steps * (self.end_value - self.initial_value) + self.initial_value
        self.validate()
        return self._value

    def get_value(self):
        """
        获取当前的学习率
        """
        if self.current_step == 0:
            return self.initial_value
        else:
            return self._value