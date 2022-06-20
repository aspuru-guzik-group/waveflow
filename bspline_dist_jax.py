from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np
from line_profiler_pycharm import profile

def M(x, k, i, t):
   if k == 1:
      if x >= t[i] and x < t[i+1]:
         if t[i+1] - t[i] == 0:
            return 0
         else:
            return 1 / (t[i+1] - t[i])
      else:
         return 0
   if t[i+k] - t[i] == 0:
      return 0
   else:
      return k * ( (x - t[i]) * M(x, k-1, i, t) + (t[i+k] - x) * M(x, k-1, i+1, t) ) / ( (k-1) * (t[i+k] - t[i]) )
def mspline(x, t, c, k):
   return sum(c[i] * M(x, k, i, t) for i in range(len(c)))



def I(x, k, i, t):
   j = np.searchsorted(t, x, 'right') - 1
   if i > j:
      return 0
   elif i <= j - k:
      return 1
   else:
      #return np.array([(t[m+k+1] - t[m]) * M(x, k+1, m, t)/(k+1) for m in range(i, j+1)]).sum()
      return np.array([(t[m + k + 1] - t[m]) * M(x, k + 1, m, t) / (k + 1) for m in range(i, j+1)]).sum()
def ispline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k + 1) and (len(c) >= n)
   return sum(c[i] * I(x, k, i, t) for i in range(n))


def B(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 + c2
def bspline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t) for i in range(n))


def MSpline_fun():

   def init_fun(k, internal_knots, initial_params=None, cardinal_splines=True):
      internal_knots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * k).clip(min=1))
      knots = np.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * k).clip(min=1))
      n_knots = len(knots)

      if n_knots - k != len(initial_params):
         print('We need number of weights plus degree = number of knots + 2 * degree.\n'
               'We got number of weights: {}; Degree {}; Number of knots + 2*degree {}'.format(len(initial_params), k, n_knots))
         exit()

      initial_params = np.random.rand(n_knots - k)
      initial_params = initial_params / sum(initial_params)


      def apply_fun(params, x, k, knots=None):
         if knots is None:
            knots = params[1]
            params = params[0]

         return mspline(x, knots, params, k)


      apply_fun = jax.vmap(apply_fun)
      if cardinal_splines:
         return initial_params, apply_fun, knots
      else:
         initial_params = (initial_params, knots)
         return initial_params, apply_fun

   return init_fun


def test_jax_splines():
   k = 5
   internal_knots = np.linspace(0, 1, 6)
   init_fun = MSpline_fun()
   params, apply_fun, knots = init_fun(k, internal_knots)

   n_points = 1000
   xx = np.linspace(internal_knots[0] - 1, internal_knots[-1] + 1, n_points)

   fig, ax = plt.subplots()
   ax.plot(xx, apply_fun(params, xx, k, knots), label='M Spline')

   ax.grid(True)
   ax.legend(loc='best')
   plt.show()