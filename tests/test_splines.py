import unittest
import sys
import numpy
import jax.numpy as jnp

sys.path.append("../")
import splines

class TestMSplines(object):

    def test_update_knots(self):
        num_splines = 5
        degree = 3
        msplines = splines.MSplines(num_splines, degree)
        # generate splines between [0,1]
        l = 0
        r = 1
        knots = jnp.linspace(l, r, num_splines + degree, endpoint=True)
        msplines.update_knots(knots, fix_point=True)
        #print(msplines.splines0_fp)

    def test_gen_simple_knots(self):
        num_splines = 5
        degree = 2
        bounds_1= [0, 1]
        knots_1 = splines.gen_simple_knots(num_splines, degree, bounds_1)
        knots_ref_1 = jnp.array([0., 0., 0.25, 0.5, 0.75, 1.0, 1.0])
        assert jnp.all(knots_1 == knots_ref_1)

        bounds_2 = jnp.array([0., 0.1, 0.3, 0.6, 1.0])
        knots_2 = splines.gen_simple_knots(num_splines, degree, bounds_2)
        knots_ref_2 = jnp.array([0., 0., 0.1, 0.3, 0.6, 1.0, 1.0])
        assert jnp.all(knots_2 == knots_ref_2)
