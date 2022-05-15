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
            df: int, number of splines to generate
            degree: the spline is continuous up to the (degree - 1)th derivative.
        '''
        self.num_splines = num_splines
        self.degree = degree # k value in the paper

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
            self.splines0_fp = 1. / (knots[1:self.num_splines + 2] - knots[:self.num_splines+1])
        except:
            raise Exception("The knots should monotonically increase!")  # overflow

        if fix_point:
            self.fix_point = True
            self.splines = None
        else:
            self.splines= []
            for i in range(self.num_splines + 1): # NOTE added an extra function
                self.splines.append(lambda x: (x >= self.knots[i]) * (x < self.knots[i+1]) * self.splines0_fp[i])

    def kernel_fp(self, coord):
        '''
        Evaluate the degree-th order spline at fixed points x.
        Args:
            coord: 1d array, points at which the splines are evaluated, coord has to increase monotonically.
        Returns:
            2d array: shape of (num_splines, nx). The ith row is the ith spline evaluated at x.
        '''
        try:
            nx = len(coord)
        except:
            coord = np.array([coord])
            nx = 1 # coord is a number
        splines_fp = np.zeros((self.num_splines + 1, nx)) # TODO should we leave the extra row as zero or with numbers?

        # Put the first order splines into the matrix
        count_x = 0
        for i in range(self.num_splines + 1): #TODO make a test
            while (count_x < nx):
                if coord[count_x] >= self.knots[i] and coord < self.knots[i+1]:
                    splines_fp[i, count_x] = self.splines0_fp
                    count_x += 1
                else:
                    break

        for k in range(2, self.degree+1): #TODO check if +1 is needed
            # k as in eq(2) in Ramsay paper
            splines_fp_temp = np.copy(splines_fp)
            _p = k / (k - 1.0)
            for i in range(self.num_splines):
                for x in range(nx):
                    _p1 = _p /(self.knots[i+k] - self.knots[i]) # making life easier
                    _p2 = (coord[x] - self.knots[i]) * splines_fp_temp[i, x] + (self.knots[i+k] - coord[x]) * splines_fp_temp[i+1, x] #TODO check here
                    splines_fp[i, x] = _p1 * _p2

        return splines_fpp


    def kernel(self):
        pass



def ISplines():
    pass

def gen_knots():
    pass

def mono_func():
    pass