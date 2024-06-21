import numpy as np

def quadratic_programming(x):
    f = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    g = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2])
    h = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    return f, g.T, h


def qp_first_inequality_constrait(x):
    f = -x[0]
    g = np.array([-1, 0, 0])
    h = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return f, g.T, h


def qp_second_inequality_constrait(x):
    f = -x[1]
    g = np.array([0, -1, 0])
    h = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return f, g.T, h



def qp_third_inequality_constrait(x):
    f = -x[2]
    g = np.array([0, 0, -1])
    h = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return f, g.T, h



def linear_programming(x):
    f = -x[0] - x[1]
    g = np.array([-1, -1])
    h = np.array([[0, 0], [0, 0]])
    return f, g.T, h



def lp_first_inequality_constraint(x):
    f = -x[0] - x[1] + 1
    g = np.array([-1, -1])
    h = np.array([[0, 0], [0, 0]])
    return f, g.T, h



def lp_second_inequality_constraint(x):
    f = x[1] - 1
    g = np.array([0, 1])
    h = np.array([[0, 0], [0, 0]])
    return f, g.T, h



def lp_third_inequality_constraint(x):
    f = x[0] - 2
    g = np.array([1, 0])
    h = np.array([[0, 0], [0, 0]])
    return f, g.T, h



def lp_fourth_inequality_constraint(x):
    f = -x[1]
    g = np.array([0, -1])
    h = np.array([[0, 0], [0, 0]])
    return f, g.T, h

