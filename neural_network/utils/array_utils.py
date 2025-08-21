import numpy as np

def shape(ndarray: list | float) -> tuple[int, ...]:
    if isinstance(ndarray, list):
        array_size = len(ndarray)
        next_size = shape(ndarray[0])
        return array_size, *next_size
    return ()

def jagged_shape(jagged_array: list | np.ndarray | float) -> list:
    shape_tuple = []
    if isinstance(jagged_array, list):
        for array in jagged_array:
            shape_tuple.append(jagged_shape(array))
        return shape_tuple
    if isinstance(jagged_array, np.ndarray):
        return jagged_array.shape
    return shape_tuple
