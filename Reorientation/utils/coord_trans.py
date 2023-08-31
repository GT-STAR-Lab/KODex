import numpy as np
from numpy.linalg import inv

def ori_transform(current_ori, desired_ori):  
    # build a new coordinate frame that define the desired_ori as [1, 0, 0] - use desired_ori to build a new coordinate basis
    # and then convert current_ori in that new coordinate frame
    x = desired_ori
    y, z = compute_orthonormal_vector(x)
    origin_x = np.array([1, 0, 0])
    origin_y = np.array([0, 1, 0])
    origin_z = np.array([0, 0, 1])
    Q = np.zeros([3,3])
    Q[0,0] = np.dot(x, origin_x)
    Q[0,1] = np.dot(x, origin_y)
    Q[0,2] = np.dot(x, origin_z)
    Q[1,0] = np.dot(y, origin_x)
    Q[1,1] = np.dot(y, origin_y)
    Q[1,2] = np.dot(y, origin_z)
    Q[2,0] = np.dot(z, origin_x)
    Q[2,1] = np.dot(z, origin_y)
    Q[2,2] = np.dot(z, origin_z)
    return np.dot(Q, current_ori)

def ori_transform_inverse(current_ori, desired_ori):  
    x = desired_ori
    y, z = compute_orthonormal_vector(x)
    origin_x = np.array([1, 0, 0])
    origin_y = np.array([0, 1, 0])
    origin_z = np.array([0, 0, 1])
    Q = np.zeros([3,3])
    Q[0,0] = np.dot(x, origin_x)
    Q[0,1] = np.dot(x, origin_y)
    Q[0,2] = np.dot(x, origin_z)
    Q[1,0] = np.dot(y, origin_x)
    Q[1,1] = np.dot(y, origin_y)
    Q[1,2] = np.dot(y, origin_z)
    Q[2,0] = np.dot(z, origin_x)
    Q[2,1] = np.dot(z, origin_y)
    Q[2,2] = np.dot(z, origin_z)
    Q = Q.T
    return np.dot(Q, current_ori)

def compute_orthonormal_vector(desired_ori):
    # compute two orthonormal vectors wrt desired_ori(new x axis) as the new y, z axis
    # for the y axis, we will have three constraints: 1).x1v1s + x2v2 + x3v3=0, 2).v1^2+v2^2+v^3=1, 3). v1=1*v2(manually specify)
    # the z axis is uniquely defined by the x and y axis
    x1 = desired_ori[0]
    x2 = desired_ori[1]
    x3 = desired_ori[2]
    v1 = v2 = np.sqrt(1 / (2 + (x1 + x2) ** 2 / x3 ** 2))
    v3 = -(x1+x2)/x3*v2
    y = np.array([v1, v2, v3])
    z1 = x2*v3 - x3*v2
    z2 = x3*v1 - x1*v3
    z3 = x1*v2 - x2*v1
    z = np.array([z1, z2, z3]) / (z1**2+z2**2+z3**2)
    return y, z

