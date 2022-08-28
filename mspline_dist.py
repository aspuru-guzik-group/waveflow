from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np
from line_profiler_pycharm import profile
import tqdm
import ortho_splines

# def M(x, k, i, t, max_k):
#    is_superflious_node = i < (max_k - 1) or i >= len(t) - max_k
#    is_first_node = i + k <= max_k-1 or i >= len(t) - max_k
#
#    if k == 1:
#       if (x >= t[i] and x < t[i+1]) or (i >= len(t) - (max_k+1) and x >= t[i] and x <= t[i+1]):
#          if t[i+1] - t[i] == 0: #is_superflious_node:
#             return 0
#          else:
#             return 1/ (t[i+1] - t[i])
#       else:
#          return 0
#    if t[i+k] - t[i] == 0: #is_first_node:
#       return 0
#    else:
#       return k * ((x - t[i]) * M(x, k - 1, i, t, max_k) + (t[i + k] - x) * M(x, k - 1, i + 1, t, max_k)) / ((k - 1) * (t[i + k] - t[i]))
def M(x, k, i, t, max_k, n_derivatives = 0):
   if k == 1:
      if (x >= t[i] and x < t[i+1]) or (i >= len(t) - (max_k+1) and x >= t[i] and x <= t[i+1]):
         if t[i+1] - t[i] == 0:
            return 0
         else:
            if n_derivatives==0:
               return 1 / (t[i + 1] - t[i])
            else:
               return 0
      else:
         return 0
   if t[i+k] - t[i] == 0:
      return 0
   else:
      if n_derivatives == 0:
         return k * ((x - t[i]) * M(x, k - 1, i, t, max_k, n_derivatives=0) + (t[i + k] - x) * M(x, k - 1, i + 1, t, max_k, n_derivatives=0)) / ((k - 1) * (t[i + k] - t[i]))
      elif n_derivatives == 1:
         return k / ((k - 1) * (t[i + k] - t[i])) * ((x - t[i]) * M(x, k - 1, i, t, max_k, n_derivatives=n_derivatives) + (t[i + k] - x) * M(x, k - 1, i + 1, t, max_k, n_derivatives=n_derivatives) + M(x, k - 1, i, t, max_k, n_derivatives=0) - M(x, k - 1, i + 1, t, max_k, n_derivatives=0))
      else:
         return k / ((k - 1) * (t[i + k] - t[i])) * ((x - t[i]) * M(x, k - 1, i, t, max_k, n_derivatives=n_derivatives) + (t[i + k] - x) * M(x, k - 1, i + 1, t, max_k, n_derivatives=n_derivatives) + n_derivatives * (M(x, k - 1, i, t, max_k, n_derivatives=n_derivatives-1) - M(x, k - 1, i + 1, t, max_k, n_derivatives=n_derivatives-1)))

def dM(x, k, i, t, max_k):
   # WARNING: Only works for cardinal splines!
   # return k * (M(x, k - 1, i, t, k - 1) - M(x, k - 1, i + 1, t, k - 1)) / (t[i + k] - t[i])
   if i>=len(t)-2*k:
      a2 = 1 / ( ((len(t) - k) - i) / ((len(t) - k) - i - 1))
   elif i < k-1:
      a2 = (i+2)/(i+1)
   else:
      a2 = 1
   return k * ((M(x, k - 1, i, t, k-1) / (t[i + k] - t[i]) - a2 * M(x, k - 1, i + 1, t, k-1) / (t[i + k + 1] - t[i + 1])))

def mspline(x, t, c, k, n_derivatives=0):
   return sum(c[i] * M(x, k, i, t, k, n_derivatives=n_derivatives) for i in range(len(c)))


def I(x, k, i, t, max_k, n_derivatives=0):
   if x == 0.0:
      j = k
   else:
      j = np.searchsorted(t, x, 'left') - 1

   if i > j or i == len(t) - (k + 1):
      return 0
   elif i <= j - k:
      if n_derivatives == 0:
         return 1
      else:
         return 0
   else:
      return np.array([(t[m + k + 1] - t[m]) * M(x, k + 1, m, t, max_k, n_derivatives=n_derivatives) / (k + 1) for m in range(i, j+1)]).sum()
def ispline(x, t, c, k, n_derivatives=0):
   n = len(t) - k - 1
   assert (n >= k + 1) and (len(c) >= n)
   return sum(c[i] * I(x, k, i, t, k+1, n_derivatives=n_derivatives) for i in range(n))


def B(x, k, i, t, max_k, n_derivative=0):
   if n_derivative == 0:
      if k == 0:
         if t[i] <= x < t[i+1] or (i >= len(t) - (max_k+1) and x >= t[i] and x <= t[i+1]):
            return 1.0
         else:
            return 0.0

         #return 1.0 if t[i] <= x < t[i+1] else 0.0
      if t[i+k] == t[i]:
         c1 = 0.0
      else:
         c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t, max_k)
      if t[i+k+1] == t[i+1]:
         c2 = 0.0
      else:
         c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t, max_k)
      return c1 + c2
   else:
      return dB(x, k , i , t, max_k, n_derivative=n_derivative)

def bspline(x, t, c, k, n_derivative=0):
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t, k, n_derivative=n_derivative) for i in range(n))

def dB(x, k, i, t, max_k, n_derivative=1):
   return k * ( (B(x, k-1, i, t, max_k, n_derivative=n_derivative-1) / (t[i+k] - t[i]) - B(x, k-1, i+1, t, max_k, n_derivative=n_derivative-1) / (t[i+k+1] - t[i+1]) ) )

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
def test_splines(test_case):

   degree = 4
   internal_knots = np.linspace(0, 1, 13)
   # internal_knots = np.random.uniform(0,1, 9)
   # internal_knots[0] = 0
   # internal_knots = np.cumsum(internal_knots)
   # internal_knots = internal_knots / internal_knots[-1]

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

   bknots = mknots#internal_knots
   bweights = np.random.rand(len(mknots) - degree - 1) - 0.5#np.random()
   bweights = bweights / np.sqrt(sum(np.abs(bweights)))

   n_points = 1000
   xx = np.linspace(internal_knots[0], internal_knots[-1], n_points)
   dx = (xx[-1] - xx[0]) / n_points


   if test_case == 'm':

      fig, ax = plt.subplots()
      for i in range(len(mweights)):
         ys = np.array([M(x, degree, i, mknots, degree, n_derivatives=0) for x in xx])
         # dys = np.array([M(x, degree, i, mknots, degree, n_derivatives=1) for x in xx])

         ax.plot(xx, ys, label='M {}'.format(i), ls='-')
         # ax.plot(xx, dys, label='dM {}/dx analytical'.format(i), ls='-')

      ax.grid(True)
      # ax.legend(loc='best')
      plt.show()


      fig, ax = plt.subplots()
      ys = np.array([mspline(x, mknots, mweights, degree) for x in xx])
      ax.plot(xx, ys, label='M Spline')
      max_val = np.max(mweights) * len(mknots)
      s = rejection_sampling(lambda x: np.array([mspline(x_, mknots, mweights, degree) for x_ in x]), 4000, xmin=0, xmax=1,
                             ymax=max_val)
      ax.hist(np.array(s), density=True, bins=100)

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()

   elif test_case == 'i':

      fig, ax = plt.subplots()
      for i in range(len(iweights)):

         ax.plot(xx, [I(x, degree, i, iknots, degree + 1, n_derivatives=0) for x in xx], label='I {}'.format(i))
         ax.plot(xx, np.array([I(x, degree, i, iknots, degree + 1, n_derivatives=1) for x in xx]),
                 label='dI/dx analytical {}'.format(i))

      ax.grid(True)
      # ax.legend(loc='best')
      plt.show()


      fig, ax = plt.subplots()
      ax.plot(xx, np.array([ispline(x, iknots, iweights, degree, n_derivatives=0) for x in xx]), label='I Spline')

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()

   elif test_case == 't':
      def uniform_pdf(x):
         return 1

      n_dim = 1
      if n_dim == 1:
         transformed_pdf = lambda x: uniform_pdf(ispline(x, iknots, iweights, degree, n_derivatives=0)) * ispline(x, iknots, iweights, degree, n_derivatives=1)

         n_points = 2000
         dx = 1/n_points
         x_range = np.linspace(0, 1, n_points)

         ys = np.array([transformed_pdf(x) for x in x_range])
         plt.plot(x_range, ys)
         plt.show()

         print(ys.sum() * dx)
      else:
         iweights[3:-3] = iweights[3:-3]
         iweights = iweights / iweights.sum(keepdims=True)
         print(iweights)

         fig, ax = plt.subplots()
         ax.plot(xx, np.array([ispline(x, iknots, iweights, degree, n_derivatives=1) for x in xx]), label='I Spline')
         ax.grid(True)
         ax.legend(loc='best')
         plt.show()

         fig, ax = plt.subplots()
         ax.plot(xx, np.array([ispline(x, iknots, iweights, degree, n_derivatives=0) for x in xx]), label='I Spline')
         ax.grid(True)
         ax.legend(loc='best')
         plt.show()




         transformed_pdf = lambda x, y: uniform_pdf(ispline(x, iknots, iweights, degree, n_derivatives=0)) * ispline(x, iknots, iweights, degree, n_derivatives=1) * \
                                        uniform_pdf(ispline(y, iknots, iweights, degree, n_derivatives=0)) * ispline(y, iknots, iweights, degree, n_derivatives=1)

         left_grid = 0.0
         right_grid = 1.0
         n_grid_points = 100
         dx = ((right_grid - left_grid) / n_grid_points) ** 2
         x = np.linspace(left_grid, right_grid, n_grid_points)
         y = np.linspace(left_grid, right_grid, n_grid_points)

         xv, yv = np.meshgrid(x, y)
         xv, yv = xv.reshape(-1), yv.reshape(-1)
         xv = np.expand_dims(xv, axis=-1)
         yv = np.expand_dims(yv, axis=-1)
         grid = np.concatenate([xv, yv], axis=-1)
         pdf_grid = np.array([transformed_pdf(x, y) for x, y in zip(grid[:,0], grid[:,1])]).reshape(n_grid_points, n_grid_points)
         plt.imshow(pdf_grid, extent=(left_grid, right_grid, left_grid, right_grid), origin='lower')
         plt.show()

         print(pdf_grid.sum() * dx)

   elif test_case == 'b':
      basis_splines = []
      fig, ax = plt.subplots()
      for i in range(len(bweights)):
         ys = np.array([B(x, degree, i, bknots, degree, n_derivative=0) for x in xx])
         basis_splines.append(ys)
         ax.plot(xx, ys, label='M {}'.format(i), ls='-')

      ax.grid(True)
      plt.show()

      basis_splines = np.array(basis_splines)
      basis_splines = basis_splines / np.linalg.norm(basis_splines, axis=-1, keepdims=True)

      fig, ax = plt.subplots()
      ys = [np.array([bweights[i] * basis_splines[i, j] for i in range(len(bweights))]).sum() for j, x in enumerate(xx)]
      ax.plot(xx, ys, label='B', ls='-')

      ax.grid(True)
      plt.show()


      # basis_splines = ortho_splines.gram_schmidt_r2l(basis_splines.T).T
      basis_splines = ortho_splines.gram_schmidt_symm(basis_splines.T).T

      print(np.linalg.norm(basis_splines, axis=-1))
      print(basis_splines[0].sum())
      print((basis_splines[0]**2).sum())
      print(np.dot(basis_splines[3], basis_splines[4]))
      print(basis_splines.shape)



      fig, ax = plt.subplots()
      for i in range(len(bweights)):
         ys = basis_splines[i]
         ax.plot(xx, ys, label='M {}'.format(i), ls='-')

      ax.grid(True)
      plt.show()

      fig, ax = plt.subplots()
      ys = [np.array([bweights[i] * basis_splines[i, j] for i in range(len(bweights))]).sum() for j, x in enumerate(xx)]
      ax.plot(xx, ys, label='B', ls='-')

      ax.grid(True)
      plt.show()


      # fig, ax = plt.subplots()
      # ys = np.array([bspline(x, bknots, bweights, degree) for x in xx])
      # ys_squared = np.array([bspline(x, bknots, bweights, degree) for x in xx])**2
      # ax.plot(xx, ys, label='B Spline')
      # ax.plot(xx, ys_squared, label='B Spline Squared')


      ax.grid(True)
      ax.legend(loc='best')
      plt.show()







if __name__ == '__main__':
   test_splines('b')










