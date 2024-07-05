import math

def nearest_power_of_two_half(x):
    # 计算0.5 * x
    target = 0.5 * x
    # 计算对数，并四舍五入到最接近的整数
    nearest_exp = round(math.log(target, 2))
    # 计算2的nearest_exp次幂
    if nearest_exp < 0:
        nearest_exp = 0
    nearest_power = int(math.ceil(2 ** nearest_exp))

    return nearest_power

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