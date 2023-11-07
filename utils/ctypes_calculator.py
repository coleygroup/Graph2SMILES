import ctypes
import numpy as np
import numpy.ctypeslib as npct


class DistanceCalculator:
    # Declare an alias for the C type int*, equivalent to int[]
    array_1d_uint8 = npct.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    array_1d_bool = npct.ndpointer(dtype=np.bool_, ndim=1, flags="C_CONTIGUOUS")

    # Load c_func.so into my_lib; my_lib.c_func is now callable
    my_lib = npct.load_library("c_calculate", "./utils")

    my_lib.c_calculate.restype = None        # return type
    my_lib.c_calculate.argtypes = [          # args type
        array_1d_uint8, ctypes.c_int32,
        array_1d_bool,
        ctypes.c_int32
    ]

    @staticmethod
    def calculate(adjacency: np.ndarray,
                  a_length: int,
                  max_distance: int) -> np.ndarray:
        flattened_distance = np.zeros(a_length * a_length, dtype=np.uint8)
        flattened_adjacency = adjacency.ravel()

        DistanceCalculator.my_lib.c_calculate(
            flattened_distance, a_length,
            flattened_adjacency,
            max_distance
        )
        distance = flattened_distance.reshape(a_length, a_length)

        return distance
