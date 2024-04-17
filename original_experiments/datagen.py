import numpy as np


def sin1(x):
    return 4*(1-x*(np.sin(x)**2 + 0.1))


def sin2(x):
    raise NotImplementedError


def sin3(x):
    return -x*(1 + 0.1*np.sin(x))


def sin4(x):
    return -np.power(x, 3)*(1 + 0.1*np.power(np.sin(x), 2))


def gamma1(x):
    raise NotImplementedError


def gamma2(x):
    raise NotImplementedError


def gamma3(x):
    raise NotImplementedError


def gamma4(x):
    raise NotImplementedError


def gamma5(x):
    raise NotImplementedError


def gamma6(x):
    return (10 - 1)/x - 5


def double_well(x):
    return 4*(x-x**3)
