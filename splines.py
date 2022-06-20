# Generate monotonic functions with splines
# following: 
#   J. O. Ramsay, Stat. Sci. 3, 425 (1988)
#   https://www.fon.hum.uva.nl/praat/manual/spline.html
# Chong Sun <sunchong137@gmail.com>

import numpy as np


def gen_simple_knots(n_splines, degree, bounds):
    '''
    Generate a sequence of simple knots.
    Args:
        n_splines: int, the 'n' value. Number of splines to generate
        degree: int, the 'k' value. The spline is continuous up to the (degree)th derivative. This is different from M-Splines.
        bounds: list or array. The boundaries that divide the region. bounds[0] and bounds[-1] are the minimum and maximum of the region.

    Returns:
        1D array of size (n_splines + degree) storing the values of the knots.
    '''
    #bounds = jnp.asarray(bounds)
    bounds = np.asarray(bounds)
    num_bounds = len(bounds)
    assert num_bounds > 1

    num_internal = n_splines - degree 
    targ_num_bounds = num_internal + 2

    if num_bounds != targ_num_bounds: # the number of boundary points has to be greater than this number
        #bounds = jnp.linspace(bounds[0], bounds[-1], num_internal + 2, endpoint=True)
        bounds = np.linspace(bounds[0], bounds[-1], num_internal + 2, endpoint=True)
        num_bounds = targ_num_bounds

    num_knots = n_splines + degree
    #knots = jnp.empty(num_knots, dtype=jnp.float32)
    #knots = knots.at[:degree].set(bounds[0])
    #knots = knots.at[degree:n_splines].set(copy.deepcopy(bounds[1:-1]))
    #knots = knots.at[n_splines:].set(bounds[-1])
    knots = np.ones(num_knots)
    knots[:degree] *= bounds[0]
    knots[degree:n_splines] = np.copy(bounds[1:-1])
    knots[n_splines:] *= bounds[-1]
    return knots

def MSplines(x, n_splines, degree, knots=None, bounds=None, verbose=0):
    '''
    Generate M-Splines. 
    Args:
        x:           float, at where the splines are evalutated.
        n_splines:   int, the 'n' value. Number of splines to generate
        degree:      int, the 'k' value. The spline is continuous up to the
        (degree)th derivative. This is different from M-Splines.
        knots:       the knots used to define the splines.
        bounds:      the boundary points to divide the interval.    

    Returns:
        1D array of size (n_splines) storing the values of splines at point
        x.
    '''

    if verbose == "DEBUG":
        print("evaluating the splines at {}".format(x))
    
    if knots is None:
        knots = gen_simple_knots(n_splines, degree, bounds)
    n_knots = len(knots)
    assert n_knots == degree + n_splines

    # k = 1
    # Mx = 1/(t_i+1 - t_i), if t_i =< x < t_i+1, otherwise 0.
    Mx = np.zeros(n_splines) # TODO 
    #j = np.searchsorted(knots, x, 'right') - 1
    for j in range(n_knots - 1): # TODO replace with numpy.searchsorted()
        if x >= knots[j] and x < knots[j+1]:
            break
    Mx[j] = 1.0 / (knots[j+1] - knots[j])

    # k > 1
    for k in range(2, degree+1):
        Mx_last = np.copy(Mx)
        _p1 = k / (k - 1.0)

        for i in range(n_splines - 1):
            _p  = (x - knots[i]) * Mx[i] + (knots[i+k] - x) * Mx[i+1]
            if abs(_p) > 1e-10: #_p2 overflow
                _p2 = _p1 / (knots[i + k] - knots[i])
                Mx[i] = _p * _p2
            else:
                Mx[i] = 0
        # last spline
        i = n_splines - 1
        _p  = (x - knots[i]) * Mx[i] 
        if abs(_p) > 1e-10: #_p2 overflow
            _p2 = _p1 / (knots[i + k] - knots[i])
            Mx[i] = _p * _p2
        else:
            Mx[i] = 0


    return Mx


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    bounds = [0, 1]
    n_splines = 6
    degree = 3
    #knots = np.array([0, 0, 0, 0.3, 0.5, 0.6, 1., 1., 1.])
    bounds = np.array([0, 0.3, 0.5, 0.6, 1.0])
    shift = 1e-3
    L = 0.0
    R = 1.0
    mesh = np.linspace(L+shift, R-shift, 200)
    Mx = []
    for x in mesh:
        m = MSplines(x, n_splines, degree, bounds=bounds, verbose="DEBUG")
        Mx.append(m)
    Mx = np.asarray(Mx)
    for i in range(n_splines):
        plt.plot(mesh, Mx[:, i])
    plt.show()
