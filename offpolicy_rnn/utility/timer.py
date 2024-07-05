import time
import inspect
import numpy as np

class Timer:
    def __init__(self):
        self.check_points = {}
        self.points_time = {}
        self.need_summary = {}
        self.init_time = time.time()

    def reset(self):
        self.check_points = {}
        self.points_time = {}
        self.need_summary = {}

    @staticmethod
    def file_func_line(stack=1):
        frame = inspect.stack()[stack][0]
        info = inspect.getframeinfo(frame)
        return info.filename, info.function, info.lineno

    @staticmethod
    def line(stack=2, short=False):
        file, func, lineo = Timer.file_func_line(stack)
        if short:
            return f"line_{lineo}_func_{func}"
        return f"line: {lineo}, func: {func}, file: {file}"

    def register_point(self, tag=None, stack=3, short=True, need_summary=True, level=0):
        if tag is None:
            tag = self.line(stack, short)
        if False and not tag.startswith('__'):
            print(f'arrive {tag}, time: {time.time() - self.init_time}, level: {level}')
        if level not in self.check_points:
            self.check_points[level] = []
            self.points_time[level] = []
            self.need_summary[level] = set()
        self.check_points[level].append(tag)
        self.points_time[level].append(time.time())
        if need_summary:
            self.need_summary[level].add(tag)

    def register_end(self, stack=4, level=0):
        self.register_point('__timer_end_unique', stack, need_summary=False, level=level)

    def summary(self, summation=False):
        if len(self.check_points) == 0:
            return dict()
        res = {}
        for level in self.check_points:
            self.register_point('__timer_finale_unique', level=level)
            res_tmp = {}
            for ind, item in enumerate(self.check_points[level][:-1]):
                time_now = self.points_time[level][ind]
                time_next = self.points_time[level][ind + 1]
                if item in res_tmp:
                    res_tmp[item].append(time_next - time_now)
                else:
                    res_tmp[item] = [time_next - time_now]
            for k, v in res_tmp.items():
                if k in self.need_summary[level]:
                    res['period_' + k] = np.mean(v)
                    if summation:
                        res['sum_' + k] = np.sum(v)
        self.reset()
        return res


def test_timer():
    timer = Timer()
    for i in range(4):
        timer.register_point()
        time.sleep(1)
    for k, v in timer.summary().items():
        print(f'{k}, {v}')

if __name__ == '__main__':
    test_timer()