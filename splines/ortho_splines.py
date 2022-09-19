# orthogonalize splines
#
import numpy as np

# def D_update(H, k):
#     pass
#
# def get_splinet(b_spline_basis, k, n_knots):
#
#     N = int(np.ceil(np.log2((n_knots + 1) / k)))
#     d = int(k * (2 ** N - 1))
#     m = n_knots + 1 - k
#     n_U = np.floor((d - m)/2)
#     n_D = k * 2 ** N - n_knots - 1 - n_U
#
#     H = b_spline_basis @ b_spline_basis.T
#     coordinates = np.eye(d)
#     I = np.arange(d) #+ 1
#
#     for l in range(N):
#
#         # A_bar, H_bar = D_update(H[I, I], N - l)
#         # coordinates[:,I] = coordinates[:, I] @ A_bar
#         # H[I,I] = H_bar
#
#         r_index = []
#         for j in range(1, k+1):
#             for i in range(1, 2**(N-l-1) + 1):
#
#                 r_index.append( 2**l * (2*i-1)*k-k+j )
#         r_index = np.array(r_index)
#
#         I = np.concatenate(I_list)


def get_dyadic_splinet(bases_splines, degree, dyadic_N):
    intervals = [[(2 * degree * (r - 1) * 2**l, (2 * degree * r - degree) * 2**l, 2 * degree * r * 2**l) for r in range(1, 2**(dyadic_N-1-l) + 1)] for l in range(dyadic_N-1)]





def gram_schmidt_symm(imat, ovlp=None):
    '''
    Symmetriezed Gram Schmidt orthogonalization.
    '''
    ## pick the central knot that is closest to the central point
    #c = (knots[-1] - knots[0]) / 2.
    #idc = np.sum([knots < c])
    #if abs(knots[idc] - c) > abs(knots[idc - 1] - c):
    #    idc = idc - 1
    #
    mat = np.copy(imat)
    N, M = mat.shape # M is the number of vectors, N is the dimension of the vectors
    npair = int(M // 2) # number of pairs

    # check if M is odd or even
    odd = True
    if M % 2 == 0:
        odd = False
    else:
        print('Currently only even number of basises supported for orthogonalization. \n Exiting...')
        exit()

    if ovlp is None:
        ovlp = np.dot(mat.T, mat) # get overlap
    omat = np.zeros((N, M))
     
    # right matrices
    matR = np.zeros((N, 2*npair))  
    ovlpR = np.zeros((2*npair, 2*npair)) 

    # reshuffle indices
    ind_j = np.concatenate([np.arange(0, 2*npair-1, 2), np.arange(1, 2*npair, 2)])
    ind_k = np.concatenate([np.arange(M-1, M-npair-1, -1), np.arange(0, npair)])

    matR[:, ind_j] = mat[:, ind_k]
    ovlpR[:, ind_j] = ovlp[:, ind_k]
    ovlpR[ind_j, :] = ovlpR[ind_k, :] # ovlpR not ovlp, we used two steps to shuffle the overlap mat

    # left matrices
    matL = np.zeros((N, M))  
    ovlpL = np.zeros((M, M)) 

    ind_jL = np.concatenate([ind_j, np.array([M-1])])
    ind_kL = np.concatenate([np.arange(0, npair), np.arange(M-1, M-npair-1, -1), np.array([npair])])

    matL[:, ind_jL] = mat[:, ind_kL]
    ovlpL[:, ind_jL] = ovlp[:, ind_kL]
    ovlpL[ind_jL, :] = ovlpL[ind_kL, :] 

    # call schmidt orthogonalization
    matL = gram_schmidt_l2r(matL, ovlpL)
    matR = gram_schmidt_l2r(matR, ovlpR)
   
    if odd:
        omat[:, npair + 1] = matL[:, M-1]
    
    # symmetrization
    for i in range(npair):
        v1 = matL[:, 2*i]
        v2 = matR[:, 2*i]
        # h = np.dot(np.dot(v1.T, ovlp), v2)
        ov = symm_ortho2v(v1, v2)
        omat[:, i] = ov[0]
        omat[:, M-i-1] = ov[1]

    square_normalization_constant = np.sqrt(N)
    for i in range(M):
        omat[:,i] = omat[:,i] * square_normalization_constant

    return omat


def symm_ortho2v(v1, v2, ovlp=None):
    '''
    Symmetrically orthogonalize two vectors.
    input:
        v1: first vector
        v2: second vector
        ovlp: inner product of x1 and x2
    output:
        two new vectors that are symmetrically orthogonalized
    '''
    if ovlp is None:
        ovlp = np.dot(v1, v2)
    assert ovlp >= 0 and ovlp <=1

    s1 = 1. / np.sqrt(1 + ovlp)
    s2 = 1. / np.sqrt(1 - ovlp)
    a1 = 0.5 * (s1 + s2)
    a2 = 0.5 * (s1 - s2)

    o1 = a1 * v1 + a2 * v2
    o2 = a2 * v1 + a1 * v2

    return o1, o2


def gram_schmidt_l2r(imat, ovlp=None):
    '''
    Orthogonalize columns in mat with Gram Schmidt algorithm.
    The orthogonalization goes from left to right.
    input:
        imat - (N, M) matrix, M: number of vectors to orthogonalize; N:
            the dimension of the Hilbert space.
        ovlp - the matrix to store the inner products 
    '''
    mat = np.copy(imat)
    N, M = mat.shape
    if ovlp is None:
        ovlp = np.dot(mat.T, mat) # get overlap
    omat = np.zeros((N, M))
    omat[:, 0] = mat[:, 0] / np.sqrt(ovlp[0, 0])
    for i in range(M - 1):
        vec = ovlp[i, (i+1):]
        mat[:, (i+1):] -= np.outer(mat[:, i], vec)/ovlp[i, i]
        ovlp[(i+1):, (i+1):] -= np.outer(vec, vec)/ovlp[i, i]
        omat[:, i+1] = mat[:, i+1]/np.sqrt(ovlp[i+1, i+1])

    return omat
    
def gram_schmidt_r2l(imat):
    '''
    Orthogonalize columns in mat with Gram Schmidt algorithm.
    The orthogonalization goes from right to left.
    input:
        imat - (N, M) matrix, M: number of vectors to orthogonalize; N:
            the dimension of the Hilbert space.
    '''
    # mat_n = mat[:, ::-1]
    omat = gram_schmidt_l2r(imat[:, ::-1])
    return omat[:, ::-1]

def qr_decomp(mat):
    q, r = np.linalg.qr(mat)
    return q


if __name__ == "__main__":
    
    mat = np.random.rand(1000, 13)
    omat = gram_schmidt_l2r(mat)
    print(omat)

    print(np.dot(omat.T, omat))
    omat1 = gram_schmidt_r2l(mat)
    print(omat1)
    print(omat1.T.dot(omat1))

    # test symm_ortho2v
    v1 = np.random.rand(10, 2)
    #v2 = np.random.rand(10)
    v1[:,0] /= np.linalg.norm(v1[:, 0])
    v1[:,1] /= np.linalg.norm(v1[:, 1])
    #v2 /= np.linalg.norm(v2)
    ov = symm_ortho2v(v1[:, 0], v1[:, 1])
    print("norm of output vectors: {}, {}; inner product of output vectors:\
          {}".format(np.linalg.norm(ov[0]), np.linalg.norm(ov[1]), np.dot(ov[0], ov[1])))


    # test symmetrized gram-schmidt
    omat = gram_schmidt_symm(mat)
    print(omat.T.dot(omat))
