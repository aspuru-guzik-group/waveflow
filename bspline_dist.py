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
   elif i < j - k + 1:
   #elif i < j - k:
      return 1
   else:
      #return np.array([(t[m+k+1] - t[m]) * M(x, k+1, m, t)/(k+1) for m in range(i, j+1)]).sum()
      return np.array([(t[m + k + 1] - t[m]) * M(x, k + 1, m, t) / (k + 1) for m in range(i, j)]).sum()
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


# def IB(x, k, i, t):
#
#    # For now assuming cardinal splines
#    h = t[1] - t[0]
#
#    if i+k+2 == len(t):
#       return 0
#    else:
#       return h * B(x, k+1, i, t) + I(x, k, i+1, t)




#@profile
def compare_splines():


   degree = 5
   knots = np.linspace(0,1,11)
   knots = np.repeat(knots, ((knots == knots[0])*degree).clip(min=1))
   knots = np.repeat(knots, ((knots == knots[-1]) * degree).clip(min=1))
   n_knots = len(knots)

   weights = np.array([0, 0, 1, 0, 1, 1, 1, 1, 3, 2, 0, 1, 0, 0])
   weights = weights / sum(weights)
   if n_knots - degree != len(weights):
      print('We need number of weights plus degree = number of knots + 2 * degree.\n'
            'We got number of weights: {}; Degree {}; Number of knots + 2*degree {}'.format(len(weights), degree, n_knots))
      exit()

   xx = np.linspace(knots[0] - 1, knots[-1] + 1, 1000)
   dx = (xx[-1] - xx[0])/1000

   mspline_naive = np.array([mspline(x, knots, weights, degree) for x in xx])
   ispline_naive = np.array([ispline(x, knots, weights, degree) for x in xx])
   ispline_nummerical = mspline_naive.cumsum() * dx
   print(ispline_nummerical[-1])

   fig, ax = plt.subplots()
   # ax.plot(xx, mspline_naive, 'r-', lw=3, label='naive')
   # ax.plot(xx, ispline_naive, 'b-', lw=3, label='inaive')
   # ax.plot(xx, ispline_nummerical, 'g-', lw=3, label='inummerical')
   for i in range(len(weights)):
      ax.plot(xx, np.array([I(x, degree, i, knots) for x in xx]), label='M {}'.format(i))

   ax.grid(True)
   ax.legend(loc='best')

   plt.show()

compare_splines()