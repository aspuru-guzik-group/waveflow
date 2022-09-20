from heapq import heappush, heappop
from bisect import bisect, insort
import jax.numpy as np


def _get_num_inversion_count(coordinates):
    '''
    Count the number of inversions necessary to sort the array
    Args:
        coordinates: array (n_dimensions,) coordinates of unsorted array

    Returns: number of inversions neccessary for sorting

    '''
    N = len(coordinates)
    if N <= 1:
        return 0

    sortList = []
    result = 0

    # heapsort, O(N*log(N))
    for i, v in enumerate(coordinates):
        heappush(sortList, (v, i))

    x = []  # create a sorted list of indexes
    while sortList:  # O(N)
        v, i = heappop(sortList)  # O(log(N))
        # find the current minimum's index
        # the index y can represent how many minimums on the left
        y = bisect(x, i)  # O(log(N))
        # i can represent how many elements on the left
        # i - y can find how many bigger nums on the left
        result += i - y

        insort(x, i)  # O(log(N))

    return result


def get_num_inversion_count(coordinates):
    '''
    Count the number of inversions necessary to sort the array
    Args:
        coordinates: array (batch_size, n_dimensions,) coordinates of unsorted array

    Returns: (batch_size, 1) number of inversions neccessary for sorting

    '''

    return np.array([_get_num_inversion_count(coordinates[i]) for i in range(coordinates.shape[0])])

def abs2rel(coordinates):
    '''
    Converts absolute coordinates to relative coordinates, assumes that array is sorted from smallest to largest.
    The first coordinate is from the origin (assumed to be at 0) to the first position.
    Args:
        coordinates: array (batch_size, n_dimensions,) in absolute coordinates

    Returns: array (batch_size, n_dimensions,) in relative coordinates
    '''
    rel_coordinates = np.diff(coordinates, prepend=0, axis=-1)

    return rel_coordinates

def rel2abs(rel_coordinates):
    '''
    Transforms relative coordinates back to absolute coordinates, assuming 0 is the origin
    Args:
        rel_coordinates: array (batch_size, n_dimensions,) in relative coordinates

    Returns: array (batch_size, n_dimensions,) in absolute coordinates

    '''

    abs_coordinates = np.cumsum(rel_coordinates, axis=-1)

    return abs_coordinates

if __name__ == '__main__':
    coordinates = np.array([[1.0, 2.5, 2.0, -3.0],
                            [0.0, -1.5, 2.0, -3.0],
                            [0.0, 1.0, 2.0, 3.0]])
    num_inversions = get_num_inversion_count(coordinates)
    sorted_coordinates = np.sort(coordinates)
    rel_coordinates = abs2rel(sorted_coordinates)
    abs_coordinates = rel2abs(rel_coordinates)

    print(num_inversions)
    print(sorted_coordinates)
    print(rel_coordinates)
    print(abs_coordinates)
