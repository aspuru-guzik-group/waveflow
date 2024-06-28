from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np
from line_profiler_pycharm import profile
import tqdm
import splines.ortho_splines as ortho_splines

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

# def M(coordinates, k, i, t, max_k):
#    is_superflious_node = i < (max_k - 1) or i >= len(t) - max_k
#    is_first_node = i + k <= max_k-1 or i >= len(t) - max_k
#
#    if k == 1:
#       if (coordinates >= t[i] and coordinates < t[i+1]) or (i >= len(t) - (max_k+1) and coordinates >= t[i] and coordinates <= t[i+1]):
#          if t[i+1] - t[i] == 0: #is_superflious_node:
#             return 0
#          else:
#             return 1/ (t[i+1] - t[i])
#       else:
#          return 0
#    if t[i+k] - t[i] == 0: #is_first_node:
#       return 0
#    else:
#       return k * ((coordinates - t[i]) * M(coordinates, k - 1, i, t, max_k) + (t[i + k] - coordinates) * M(coordinates, k - 1, i + 1, t, max_k)) / ((k - 1) * (t[i + k] - t[i]))
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
   # return k * (M(coordinates, k - 1, i, t, k - 1) - M(coordinates, k - 1, i + 1, t, k - 1)) / (t[i + k] - t[i])
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


def B(x, k, i, t, max_k, n_derivatives=0):
   if n_derivatives == 0:
      if k == 0:
         if t[i] <= x < t[i+1] or (i >= len(t) - (max_k+2) and x >= t[i] and x <= t[i+1]):
            return 1.0
         else:
            return 0.0

         #return 1.0 if t[i] <= coordinates < t[i+1] else 0.0
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
      return dB(x, k, i, t, max_k, n_derivative=n_derivatives)

def bspline(x, t, c, k, n_derivative=0):
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t, k, n_derivatives=n_derivative) for i in range(n))

def dB(x, k, i, t, max_k, n_derivative=1):
   if t[i+k] - t[i] == 0:
      c1 = 0
   else:
      c1 = B(x, k - 1, i, t, max_k, n_derivatives=n_derivative - 1) / (t[i + k] - t[i])
   if t[i+k+1] - t[i+1] == 0:
      c2 = 0
   else:
      c2 = B(x, k - 1, i + 1, t, max_k, n_derivatives=n_derivative - 1) / (t[i + k + 1] - t[i + 1])

   return k * ( c1 - c2 )





# @profile
def test_splines(test_case):
   degree = 5
   cardinal_basis = True
   if cardinal_basis:
      n_internal_knots = 10  # degree * 2**dyadic_N - 1
      internal_knots = np.linspace(0, 1, n_internal_knots)
   else:
      internal_knots = np.random.uniform(0, 1, 9)
      internal_knots[0] = 0
      internal_knots = np.cumsum(internal_knots)
      internal_knots = internal_knots / internal_knots[-1]

   n_points = 1000
   xx = np.linspace(internal_knots[0], internal_knots[-1], n_points)
   dx = (xx[-1] - xx[0]) / n_points


   if test_case == 'm':
      mknots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * degree).clip(min=1))
      mknots = np.repeat(mknots, ((mknots == mknots[-1]) * degree).clip(min=1))
      mweights = np.random.rand(len(mknots) - degree)
      mweights[0] = 0
      mweights[-1] = 0
      mweights = mweights / sum(mweights)

      fig, ax = plt.subplots()
      for i in range(len(mweights)):
         ys = np.array([M(x, degree, i, mknots, degree, n_derivatives=0) for x in xx])
         # dys = np.array([M(coordinates, degree, i, mknots, degree, n_derivatives=1) for coordinates in xx])

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
      np.random.seed(3)
      iknots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * (degree + 1)).clip(min=1))
      iknots = np.repeat(iknots, ((iknots == iknots[-1]) * (degree + 1)).clip(min=1))
      iweights = np.random.rand(len(iknots) - degree)
      iweights[0] = 0
      iweights[-1] = 0
      iweights = iweights / sum(iweights)

      fig, ax = plt.subplots()
      for i in range(len(iweights)):

         ax.plot(xx, [I(x, degree, i, iknots, degree + 1, n_derivatives=0) for x in xx], label='I {}'.format(i))
         # ax.plot(xx, np.array([I(x, degree, i, iknots, degree + 1, n_derivatives=1) for x in xx]),
         #         label='dI/dx analytical {}'.format(i))

      ax.grid(True)
      # ax.legend(loc='best')
      plt.tight_layout()
      plt.savefig('./../figures/isplines.pdf')
      # plt.show()


      fig, ax = plt.subplots()
      ax.plot(xx, np.array([ispline(x, iknots, iweights, degree, n_derivatives=0) for x in xx]), label='I Spline')

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()



   elif test_case == 'b':
      np.random.seed(3)
      bknots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * degree + 1).clip(min=1))
      bknots = np.repeat(bknots, ((bknots == bknots[-1]) * degree + 1).clip(min=1))
      bweights = np.random.rand(len(bknots) - degree - 1) - 0.5  # np.random()
      bweights[:1] = 0
      bweights[-1] = 0
      bweights = bweights / np.sqrt(sum(bweights ** 2))


      basis_splines = []
      d_basis_splines = []

      for i in range(len(bweights)):
         ys = np.array([B(x, degree, i, bknots, degree, n_derivatives=0) for x in xx])
         basis_splines.append(ys)
         dys = np.array([B(x, degree, i, bknots, degree, n_derivatives=1) for x in xx])
         d_basis_splines.append(dys)


      basis_splines = np.array(basis_splines)
      # basis_splines = basis_splines / np.sqrt((basis_splines**2).T.sum(-1))

      o_basis_splines = ortho_splines.gram_schmidt_symm(basis_splines.T).T
      o_basis_splines = o_basis_splines / np.sqrt((o_basis_splines**2).sum(-1)[:,None] / n_points)
      basis_change_matrix_b_to_ob = o_basis_splines @ np.linalg.pinv(basis_splines)
      basis_change_matrix_ob_to_b = basis_splines @ np.linalg.pinv(o_basis_splines)

      fig, ax = plt.subplots()
      for i in range(len(bweights)):
         ax.plot(xx, basis_splines[i])
      ax.grid(True)
      plt.tight_layout()
      plt.savefig('./../figures/bsplines.pdf')

      fig, ax = plt.subplots()
      for i in range(len(bweights)):
         ax.plot(xx, o_basis_splines[i])
      ax.grid(True)
      plt.tight_layout()
      plt.savefig('./../figures/obsplines.pdf')



      #bweights = np.random.rand(len(bknots) - degree - 1) - 0.5  # np.random()
      # bweights = bweights / np.sqrt(sum(bweights ** 2))
      obweights = bweights @ basis_change_matrix_ob_to_b
      obweights = obweights / np.sqrt(sum(obweights ** 2))



      print('Orthogonality ', np.dot(o_basis_splines[3], o_basis_splines[4]))
      print('Square integral ', ((o_basis_splines[4]**2) * 1/o_basis_splines[4].shape[0]).sum())
      # ob_splines = ortho_splines.get_splinet(basis_splines, degree, len(bknots))

      # fig, ax = plt.subplots()
      # ys = np.array([bspline(coordinates, bknots, bweights, degree, n_derivatives=0) for coordinates in xx])
      # ax.plot(xx, ys)
      #
      # plt.show()

      # fig, ax = plt.subplots()
      # for i in range(len(bweights)):
      #    ys = np.array([B(coordinates, degree, i, bknots, degree, n_derivatives=2) for coordinates in xx])
      #    ax.plot(xx, ys, label='OB {}'.format(i), ls='-')
      # plt.show()


      # fig, ax = plt.subplots()
      # for i in range(len(bweights)):
      #    ys = o_basis_splines[i]
      #    ax.plot(xx, ys, label='OB {}'.format(i), ls='-')
      #
      # ax.grid(True)
      # plt.show()




      def obspline(x, n_derivative=0):
         bspline_vec = np.array([np.array([B(x_, degree, i, bknots, degree, n_derivatives=n_derivative) for i in range(len(bweights))]) for x_ in x]).T
         ob_spline_vec = basis_change_matrix_b_to_ob @ bspline_vec
         return obweights.dot(ob_spline_vec), np.sqrt((bspline_vec**2).T.sum(-1))

      def obspline_squared(x):
         bspline_vec = np.array([np.array([B(x_, degree, i, bknots, degree, n_derivatives=0) for i in range(len(bweights))]) for x_ in x]).T
         ob_spline_vec = basis_change_matrix_b_to_ob @ bspline_vec
         return obweights.dot(ob_spline_vec)**2

      oys, N = obspline(xx, n_derivative=0)
      ys = np.array([bspline(x, bknots, obweights @ basis_change_matrix_b_to_ob, degree, n_derivative=0) for x in xx])


      bobweights = obweights @ basis_change_matrix_b_to_ob
      max_val = (bobweights**2).max()
      print('Estimated max val upper bound ', max_val)
      print('Real max val ', np.max(oys ** 2))
      print('Square integral over entire spline', (oys ** 2 * 1 / n_points).sum())
      print('Square integral over entire spline', (ys ** 2 * 1 / n_points).sum())
      print('\n\n')

      fig, ax = plt.subplots()
      ax.plot(xx, ys, label='B', ls='-')
      # ax.plot(xx, oys, label='OB', ls='-')
      # ax.plot(xx, oys ** 2, label='B_Squared', ls='-')
      # s = rejection_sampling(obspline_squared, 1000, xmin=0, xmax=1, ymax=max_val)
      # ax.hist(np.array(s), density=True, bins=100)

      ax.grid(True)
      #plt.show()
      plt.tight_layout()
      plt.savefig('./../figures/bsplinecurve.pdf')




      d_o_basis_splines = basis_change_matrix_b_to_ob @ d_basis_splines

      # for i in range(len(bweights)):
      #    fig, ax = plt.subplots()
      #    ys = np.array([np.array(o_basis_splines[i, j]) for j, coordinates in enumerate(xx)])
      #    dys_n = np.gradient(ys, 1 / n_points, edge_order=2)
      #    dys = np.array([np.array(np.array(d_o_basis_splines)[i, j]) for j, coordinates in enumerate(xx)])
      #    ax.plot(xx, dys_n, label='del B n', ls='-')
      #    ax.plot(xx, dys, label='del B', ls='-')
      #    ax.legend()
      #    plt.show()


      # fig, ax = plt.subplots()
      # ys = np.array([np.array([bweights[i] * o_basis_splines[i, j] for i in range(len(bweights))]).sum() for j, coordinates in
      #                enumerate(xx)])
      # dys_n = np.gradient(ys, 1/n_points, edge_order=2)
      #
      # dys = np.array([np.array([bweights[i] * d_o_basis_splines[i, j] for i in range(len(bweights))]).sum() for j, coordinates in
      #                enumerate(xx)])
      # ax.plot(xx, dys_n, label='del B n', ls='-')
      # ax.plot(xx, dys, label='del B', ls='-')
      # ax.legend()
      # plt.show()




if __name__ == '__main__':
   test_splines('i')










