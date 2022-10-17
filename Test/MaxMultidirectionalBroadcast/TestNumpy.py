import numpy as np

a = np.array([ 0.35974154, -2.20873388,  0.95746274]).reshape(1, 3, 1)
b = np.array([ 0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695]).reshape(2, 3, 1)
c = np.array([-0.41468981, -0.46591926,  0.56172534,  0.05616931]).reshape(1, 1, 4)

common_shape = (2, 3, 4)
a_b = np.broadcast_to(a, common_shape)
print("a_b = ", a_b.flatten())
b_b = np.broadcast_to(b, common_shape)
print("b_b = ", b_b.flatten())
c_b = np.broadcast_to(c, common_shape)
print("c_b = ", c_b.flatten())

y = [max(_a, _b,_c) for (_a,_b,_c) in zip(a_b.flatten(), b_b.flatten(), c_b.flatten())]

print("a = ", a.flatten())
print("b = ", b.flatten())
print("c = ", c.flatten())
print("max(a,b,c) = ", np.array(y))

# 0.7590198503375636, 0.7590198503375636, 0.7590198503375636, 0.7590198503375636, -0.41468980634400543, -0.465444611539521, 0.5617253354820355, 0.05616930535561424, 0.9574627354854222, 0.9574627354854222, 0.9574627354854222, 0.9574627354854222, 0.3597415448611981, 0.3597415448611981, 0.5617253354820355, 0.3597415448611981, 0.08269051091686609, 0.08269051091686609, 0.5617253354820355, 0.08269051091686609, 0.9574627354854222, 0.9574627354854222, 0.9574627354854222, 0.9574627354854222
