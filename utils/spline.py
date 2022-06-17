# Generate monotonic functions with splines
# following: J. O. Ramsay, Stat. Sci. 3, 425 (1988)
# Chong Sun <sunchong137@gmail.com>

# TODO replace with jax?
import numpy as np

class MSplines(object):
    '''
    Generates the basis of M-Splines and corresponding integrals and derivatives.
    '''
    def __init__(self, num_splines, degree=3):
        '''
        Args:
            num_splines: int, number of splines to generate
            degree: the spline is continuous up to the (degree - 1)th derivative.
        '''
        self.num_splines = num_splines
        self.degree = degree # degree value in the paper

    def update_knots(self, knots, fix_point=False):
        '''
        Args:
            knots: sorted 1d array, the internal breakpoints that define the splines.
            fix_point: if True, evaluate the splines at fixed points; else, also generate a list of functions.
        '''
        self.knots = np.asarray(knots)
        self.bounds = (knots[0], knots[-1]) # lower and upper points, should be the two ends of the interval
        self.nknots = self.num_splines + self.degree

        if self.nknots != len(self.knots): # TODO is this necessary?
            print("WARNING", "Descarding the last :{d} knots!")
            self.knots = self.knots[:self.nknots]

        # generate the first order M-splines (uniform distribution)
        try:
            self.splines0_fp = 1. / (knots[1:self.num_splines + 1] - knots[:self.num_splines])
        except:
            raise Exception("The knots should monotonically increase!")  # overflow

        if fix_point:
            self.fix_point = True
            self.splines = None
        else:
            self.splines= []
            for i in range(self.num_splines): # NOTE added an extra function
                self.splines.append(lambda x: (x >= self.knots[i]) * (x < self.knots[i+1]) * self.splines0_fp[i])

    def kernel_fp(self, coord):
        '''
        Evaluate the degree-th order spline at fixed points x.
        Args:
            coord: 1d array, points at which the splines are evaluated, coord has to increase monotonically.
        Returns:
            2d array: shape of (num_splines, nx). The ith row is the ith spline evaluated at x.
            list of lists: the coordinates are separated according to the knots
        '''
        try:
            nx = len(coord)
        except:
            coord = np.array([coord])
            nx = 1 # coord is a number
        splines_fp = np.zeros((self.num_splines, nx)) # store the values of the splines at coord as a matrix

        # Put the first order splines into the matrix
        count_x = 0
        group_x = [] # group x values according to the knots.
        for i in range(self.num_splines): #TODO make a test, return the groups of x so that we can use it for I splines.
            group_x.append([])
            while (count_x < nx):
                if coord[count_x] >= self.knots[i] and coord < self.knots[i+1]:
                    group_x[i].append(coord[count_x])
                    splines_fp[i, count_x] = self.splines0_fp
                    count_x += 1
                else:
                    break

        for k in range(2, self.degree+1): #TODO check if +1 is needed
            # degree as in eq(2) in Ramsay paper
            splines_fp_temp = np.copy(splines_fp)
            _p = k / (k - 1.0)
            for i in range(self.num_splines):
                _p1 = _p / (self.knots[i + k] - self.knots[i])
                for x in range(nx):
                    _p2 = (coord[x] - self.knots[i]) * splines_fp_temp[i, x] + (self.knots[i+k] - coord[x]) * splines_fp_temp[i+1, x] #TODO check here
                    splines_fp[i, x] = _p1 * _p2

        return splines_fp, group_x

    def kernel(self):
        '''
        Returns: a list of functions that encode the splines.
        TODO check if the list of functions even work.
        '''
        for k in range(2, self.degree):
            splines_temp = self.splines.copy()
            self.splines = []
            _p = k / (k - 1.0)
            for i in range(self.num_splines - 1):
                _p1 = _p / (self.knots[i + k] - self.knots[i])
                func1 = lambda x: x - self.knots[i]
                func2 = lambda x: self.knots[k+i] - x
                self.splines.append(lambda x: _p1 * (splines_temp[i](x) * func1 + splines_temp[i+1](x) * func2))

            # last spline
            _p1 = _p / (self.knots[self.num_splines + k - 1] - self.knots[self.num_splines - 1])
            func1 = lambda x: x - self.knots[self.num_splines - 1]
            self.splines.append(lambda x: _p1 * splines_temp[i](x) * func1)

        return self.splines


def isplines(coord, num_splines, knots, degree=3, msplines=None):
    '''
    I-splines from a simplified form. Following Eq. (5) in Ramsay's paper.
    Constraints: Only one knot at each interior boundary.
    Note: this requires M-Splines with degree of (degree+1).
    Args:
        coord: 1d array, points at which the splines are evaluated, coord has to increase monotonically.
        num_splines: int, number of splines to generate
        degree: the spline is continuous up to the (degree)th derivative. This is different from M-Splines.
        msplines: 2d array of m-splines at coord.
    Returns:
        2d array: shape of (num_splines, nx). The ith row is the ith spline evaluated at x.
    '''
    try:
        nx = len(coord)
    except:
        coord = np.array([coord])
        nx = 1  # coord is a number
    assert len(knots) >= num_splines + degree + 1
    # generate M-Splines if not provided
    if msplines is None:
        msplines_obj = MSplines(num_splines=num_splines, degree=degree+1) # for M splines one needs an extra degree
        msplines_obj.update_knots(knots, fix_point=True)
        msplines, group_x = msplines_obj.kernel_fp(coord)
    assert nx == msplines.shape[-1]
    assert len(group_x) == num_splines # see constraints

    # get I splines
    splines_fp = np.zeros((num_splines, nx))
    # evaluate the prefactors
    pre_fact = knots[(degree+1):(num_splines+degree+1)] - knots[:num_splines]
    pre_fact /= (degree + 1)
    count_x = 0
    for ix in range(num_splines): # TODO make a test
        l = len(group_x[ix])
        min_j = max(0, ix - degree + 1)
        splines_fp[:min_j, count_x:(count_x+l)] = 1
        for j in range(min_j, ix+1):
            splines_fp[j, count_x:(count_x+l)] = np.dot(pre_fact[j:ix+1], msplines[j:ix+1, count_x:(count_x+l)])
        count_x += l

    return splines_fp

def isplines_int(coord, num_splines, knots, degree=3, msplines=None):
    '''
    Integration of M-Splines, returning a monotonic function. Following Eq. (4) in Ramsay's paper.
    Args:
        coord: 1d array, points at which the splines are evaluated, coord has to increase monotonically.
        num_splines: int, number of splines to generate
        degree: the spline is continuous up to the (degree)th derivative. This is different from M-Splines.
        msplines: 2d array of m-splines at coord.
    Returns:
        2d array: shape of (num_splines, nx). The ith row is the ith spline evaluated at x.
    '''
    try:
        nx = len(coord)
    except:
        coord = np.array([coord])
        nx = 1  # coord is a number
    #assert len(knots) >= num_splines + degree + 1
    # generate M-Splines if not provided
    if msplines is None:
        msplines_obj = MSplines(num_splines=num_splines, degree=degree)
        msplines_obj.update_knots(knots, fix_point=True)
        msplines, _ = msplines_obj.kernel_fp(coord)
    assert nx == msplines.shape[-1]
    #assert len(group_x) == num_splines  # see constraints

    splines_fp = np.zeros((num_splines, nx))
    # start integration
    diff = coord[1:] - coord[:-1]
    for ix in range(1, nx):
        splines_fp[:, ix] = splines_fp[:, ix-1] + msplines[:, ix] * diff[ix-1]

    return splines_fp   

def gen_knots():
    pass

def mono_func():
    pass