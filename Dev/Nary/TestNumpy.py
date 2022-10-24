import numpy as np
import math

def opfn(name, a, b, c):
    if (name == "Max"):
        return max(a, b,c)
    elif (name == "Min"):
        return min(a,b,c)
    elif (name == "Sum"):
        return a + b + c
    elif (name == "Mean"):
        return (a + b + c) / 3.

def test_nary(name):
    a = np.array([ 0.35974154, -2.20873388,  0.95746274]).reshape(1, 3, 1)
    b = np.array([ 0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695]).reshape(2, 3, 1)
    c = np.array([-0.41468981, -0.46591926,  0.56172534,  0.05616931]).reshape(1, 1, 4)

    common_shape = (2, 3, 4)
    a_b = np.broadcast_to(a, common_shape)
    b_b = np.broadcast_to(b, common_shape)
    c_b = np.broadcast_to(c, common_shape)

    y = np.array([opfn(name, _a, _b,_c) for (_a,_b,_c) in zip(a_b.flatten(), b_b.flatten(), c_b.flatten())])

    print("a = ", a.flatten())
    print("b = ", b.flatten())
    print("c = ", c.flatten())
    print(f"{name}(a,b,c) = {y}\n")

test_nary("Max")
test_nary("Min")
test_nary("Mean")
test_nary("Sum")

