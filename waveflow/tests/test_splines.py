import unittest
import sys
import numpy as np
#import jax.numpy as jnp

sys.path.append("../")
import splines

def test_gen_simple_knots():
        num_splines = 5
        degree = 2
        bounds_1= [0, 1]
        knots_1 = splines.gen_simple_knots(num_splines, degree, bounds_1)
        #knots_ref_1 = jnp.array([0., 0., 0.25, 0.5, 0.75, 1.0, 1.0])
        #assert jnp.all(knots_1 == knots_ref_1)
        knots_ref_1 = np.array([0., 0., 0.25, 0.5, 0.75, 1.0, 1.0])
        assert np.linalg.norm(knots_1 - knots_ref_1) < 1e-10

        bounds_2 = np.array([0., 0.1, 0.3, 0.6, 1.0])
        knots_2 = splines.gen_simple_knots(num_splines, degree, bounds_2)
        #knots_ref_2 = jnp.array([0., 0., 0.1, 0.3, 0.6, 1.0, 1.0])
        #assert jnp.all(knots_2 == knots_ref_2)
        knots_ref_2 = np.array([0., 0., 0.1, 0.3, 0.6, 1.0, 1.0])
        print(knots_2)
        print(knots_2 - knots_ref_2)
        assert np.linalg.norm(knots_2 - knots_ref_2) < 1e-10
