import numpy as np
from collections.abc import Iterable
import copy

# Find the number of dimensions in a nested list
def find_n_dim(array, depth=0):
    if isinstance(array, Iterable):
        if len(array)==0:
            return depth
        return max([find_n_dim(e, depth+1) for e in array])
    return depth

# For each scalar, list the length of his parents
def list_shapes(array, shape=[]):
    if isinstance(array, Iterable):
        if len(array)==0:
            return [shape+[0]]
        shape.append(len(array))
        nested_shapes = []
        for e in array:
            nested_shapes += list_shapes(e, copy.deepcopy(shape))
        return nested_shapes
    return [shape+[1]]

# Give a shape where the size of a given dimension is the largest find in the nested lists
def find_max_shapes(array):
    n_dim = find_n_dim(array)
    shapes = list_shapes(array, shape=[])
    max_shapes = [0 for _ in range(n_dim)]
    for d in range(n_dim):
        for s in shapes:
            if len(s)>=d:
                if s[d] > max_shapes[d]:
                    max_shapes[d] = s[d]
    return tuple(max_shapes)

# Cast nested lists in ndarray
def fill_dims(array, value=0):
    max_shapes = find_max_shapes(array)
    r = np.ones(shape=max_shapes)*value
    for coord, _ in np.ndenumerate(r):
        a = array
        find = True
        for idx in coord:
            if isinstance(a, Iterable):
                if len(a)>idx:
                    a = a[idx]
                else:
                    # a dimension of a is not large enought then:
                    # r[coord] = 0
                    find = False
                    break
            else:
                # r has more dimensions than a then:
                # r[i0,...,in,j0,...,jm] = a[i0,...,in]
                break
        if find:
            r[coord] = a
    return r
            
            
        