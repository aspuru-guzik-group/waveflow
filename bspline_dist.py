from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np
from line_profiler_pycharm import profile
import tqdm

def M(x, k, i, t, max_k):
   is_superflious_node = i < (max_k - 1) or i >= len(t) - max_k
   is_first_node = i + k <= max_k-1 or i >= len(t) - max_k
   if not is_superflious_node and t[i+1] - t[i] == 0:
      print(i)
   if is_superflious_node and not t[i+1] - t[i] == 0:
      print(i)
   if t[i+k] - t[i] == 0 and not is_first_node:
      print(i)
   if is_first_node and not t[i+k] - t[i] == 0:
      print(i)

   if k == 1:
      if x >= t[i] and x < t[i+1]:
         if is_superflious_node:#t[i+1] - t[i] == 0:
            return 0
         else:
            return 1 / (t[i+1] - t[i])
      else:
         return 0
   if is_first_node:#t[i+k] - t[i] == 0:
      return 0
   else:
      # M_k_minus_one_i = cache['{};{}'.format(k - 1, i)] if '{};{}'.format(k - 1, i) in cache else M(x, k - 1, i, t, cache)
      # M_k_minus_one_i_plus_one = cache['{};{}'.format(k - 1, i + 1)] if '{};{}'.format(k - 1, i + 1) in cache else M(x, k - 1, i + 1, t, cache)

      # return k * ( (x - t[i]) * M_k_minus_one_i + (t[i+k] - x) * M_k_minus_one_i_plus_one ) / ( (k-1) * (t[i+k] - t[i]) )
      return k * ((x - t[i]) * M(x, k - 1, i, t, max_k) + (t[i + k] - x) * M(x, k - 1, i + 1, t, max_k)) / ((k - 1) * (t[i + k] - t[i]))
def mspline(x, t, c, k):
   return sum(c[i] * M(x, k, i, t, k) for i in range(len(c)))



def I(x, k, i, t, max_k):
   j = np.searchsorted(t, x, 'right') - 1
   if i > j:
      return 0
   elif i <= j - k:
      return 1
   else:
      #return np.array([(t[m+k+1] - t[m]) * M(x, k+1, m, t)/(k+1) for m in range(i, j+1)]).sum()
      return np.array([(t[m + k + 1] - t[m]) * M(x, k + 1, m, t, max_k) / (k + 1) for m in range(i, j+1)]).sum()
def ispline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k + 1) and (len(c) >= n)
   return sum(c[i] * I(x, k, i, t, k+1) for i in range(n))


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


def rejection_sampling(function, num_samples, xmin=-10, xmax=10, ymax=1):
   x = np.random.uniform(low=xmin, high=xmax, size=num_samples * 4)
   y = np.random.uniform(low=0, high=ymax, size=num_samples * 4)
   passed = (y < function(x)).astype(bool)
   all_x = x[passed]

   full_batch = False
   if all_x.shape[0] > num_samples:
      full_batch = True

   while not full_batch:
      x = np.random.uniform(low=xmin, high=xmax, size=num_samples)
      y = np.random.uniform(low=0, high=ymax, size=num_samples)
      passed = (y < function(x)).astype(bool)
      all_x = np.concatenate([all_x, x[passed]])

      if all_x.shape[0] > num_samples:
         full_batch = True

   return all_x[:num_samples]



# @profile
def test_splines():

   degree = 3
   internal_knots = np.linspace(0, 1, 8)

   mknots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * degree).clip(min=1))
   mknots = np.repeat(mknots, ((mknots == mknots[-1]) * degree).clip(min=1))
   mweights = np.random.rand(len(mknots) - degree)
   mweights[0] = 0
   mweights[-1] = 0
   mweights = mweights / sum(mweights)

   iknots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * (degree + 1)).clip(min=1))
   iknots = np.repeat(iknots, ((iknots == iknots[-1]) * (degree + 1)).clip(min=1))
   iweights = np.random.rand(len(iknots) - degree)
   iweights[0] = 0
   iweights[-1] = 0
   iweights = iweights / sum(iweights)

   n_points = 100
   xx = np.linspace(internal_knots[0] - 1, internal_knots[-1] + 1, n_points)
   dx = (xx[-1] - xx[0]) / n_points



   fig, ax = plt.subplots()
   # for i in range(len(mweights)):
   #    ax.plot(xx, np.array([M(x, degree, i, mknots, degree) for x in xx]), label='M {}'.format(i))
   #
   # ax.plot(xx, np.array([mspline(x, mknots, mweights, degree) for x in xx]), label='M Spline')

   max_val = np.max(mweights) * len(mknots)
   # for i in tqdm.tqdm(range(1000)):
   #    s = rejection_sampling(lambda x: np.array([mspline(x_, mknots, mweights, degree) for x_ in x]), 256, xmin=0, xmax=1, ymax=max_val)
   # s = rejection_sampling(lambda x: np.array([mspline(x_, mknots, mweights, degree) for x_ in x]), 256, xmin=0, xmax=1,
   #                        ymax=max_val)
   # ax.hist(np.array(s), density=True, bins=100)

   # ax.grid(True)
   # ax.legend(loc='best')
   # plt.show()

   fig, ax = plt.subplots()
   # for i in range(len(iweights)):
   #    ax.plot(xx, np.array([I(x, degree, i, iknots, degree+1) for x in xx]), label='I naive {}'.format(i))

   i = 4
   ax.plot(xx, np.array([I(x, degree, i, iknots, degree + 1) for x in xx]), label='I naive {}'.format(i))

   # ax.plot(xx, np.array([ispline(x, iknots, iweights, degree) for x in xx]), label='I Spline')

   ax.grid(True)
   ax.legend(loc='best')
   plt.show()




test_splines()










