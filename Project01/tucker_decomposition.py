from argparse import ArgumentParser
import scipy.io as sio
import numpy as np
import sys
import time


# Unfold the tensor.
#
# ** Warning **
# The return matrix shares the same memory as the input tensor. Modify the
# return tensor will also modify the input tensor.
def tensor_unfold(tensor, mode):
    h, w, c = tensor.shape  # @h: hight, @w: width, @c: channels
    if mode == 0:
        return tensor.reshape((h, c*w))
    elif mode == 1:
        return tensor.reshape((h*w, c)).reshape((w, h*c), order='F')
    else:
        return tensor.reshape((h*w*c, 1)).reshape((c, w*h), order='F')


# Fold the tensor.
def _tensor_fold(tensor, shape, mode):
    if mode == 0:
        return tensor.reshape(shape)
    else:
        mode_dim = shape.pop(mode)
        shape.insert(0, mode_dim)
        if mode == 1:
            return np.moveaxis(
                tensor.reshape(shape, order='F'), 0, mode)
        else:
            return np.moveaxis(
                tensor.reshape(shape), 0, mode)


# N-mode product of a tensor by a matrix.
def n_mode_product(tensor, matrix, mode):
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return _tensor_fold(
        np.matmul(matrix, tensor_unfold(tensor, mode)),
        new_shape,
        mode)


# Load arguments.
parser = ArgumentParser()
parser.add_argument(
    '--rank',
    help='The rank of the approximate data: rank-(P, Q, R)\n'
        +'Please seperate P, Q and R with comma but no spaces between them.',
    dest='rank',
)
parser.add_argument(
    '--path',
    help='The path of data.mat.',
    dest='pth'
)

args = parser.parse_args()
if args.rank is None or args.pth is None:
    print('Error!! Expect 2 arguments: `rank` and `path`!!\n')
    parser.print_help()
    sys.exit(0)

# Store data to a ndarray.
# @data.shape = 96 x 64 x 8
data = sio.loadmat(args.pth)['data']

t = [0, 0]
##############################################################################
# Problem 01.
#
# Get the approximate data by minimizing A, B, C alternatively. For example, 
# we fix B and C. Multiply the origin data by the pseudo inverse of the nth
# mode product of B and C.
#
# @approx_data: The best rank-(P, Q, R) approximation of @data.
# @G: The core tensor.
# @U: @U[0], @U[1], @U[2] Contain P, Q, R columns, respectively. Use them and
#   the core tensor to approximate @data.
# @R: A list of rank of @U: [P, Q, R].
##############################################################################

# Step 01.
#
# Initialization: Fill A, B, C with random values. Then normalize them.
approx_data = None
R = [int(x) for x in args.rank.split(',')] # Rank-(P, Q, R)
G = np.random.rand(*R)
U = [np.random.rand(data.shape[i], R[i]) for i in range(data.ndim)]
U = [u/np.linalg.norm(u, axis=0) for u in U]    # Normalize @U.

# Step 02.
#
# Iterate until convergence, or until it iterates 100 times.

# @prev_approx_data: The approximate data we got from last iteration. We
# compare it with the approximate data we get now, in order to check that
# whether it converges.
prev_approx_data = np.zeros(data.shape)

print('Problem 01')
print('iterations\terror')
t_start = time.time()
for iters in range(100):
    # Update @U[0] ~ @U[2].
    for i in range(data.ndim):
        if i == 0:
            tmp = np.kron(U[1], U[2]).T
        elif i == 1:
            tmp = np.kron(U[2], U[0]).T
        else:
            tmp = np.kron(U[0], U[1]).T
        tmp = np.linalg.pinv(np.matmul(tensor_unfold(G, i), tmp), rcond=1e-7)
        tmp = np.matmul(tensor_unfold(data, i), tmp)
        U[i] = tmp / np.linalg.norm(tmp, axis=0)
    # Update @G. Then we can construct the approximate data.
    tmp = data
    for i in range(data.ndim):
        tmp = n_mode_product(tmp, np.linalg.pinv(U[i], rcond=1e-7), i)
    G = tmp
    for i in range(data.ndim):
        tmp = n_mode_product(tmp, U[i], i)
    approx_data = tmp
    # @err: The error between the origin data and the approximate data.
    err = data - approx_data
    print('\t%d\t%.6lf' % (iters, np.linalg.norm(err)))
    # Break if it converges.
    if np.allclose(approx_data, prev_approx_data, 1e-5, 1e-6):
        break
    prev_approx_data = approx_data
t[0] += time.time() - t_start


##############################################################################
# Problem 02.
#
# Implement Higher-Order Orthogonal Iteration (HOOI)
##############################################################################

# Step 01.
#
# Initialization: Set A, B, C with the left singular subspace of @data(n),
# where @data(n) is matrix unfolding:
#       I1 I2 ... In ... IN -> In X (I1 I2 ... In-1 In+1 ... IN)
U = []
for i in range(data.ndim):
    U.append(
        np.linalg.svd(
            tensor_unfold(data, i),
            full_matrices=False
        )[0][:R[i], :].T
    )

# Step 02.
#
# Iterate until convergence, or until it iterates 100 times.

print('\nProblem 02')
print('iterations\terror')
prev_approx_data = np.zeros(data.shape)
t_start = time.time()
for iters in range(100):
    tmp_u = None
    # Update @U[0] ~ @U[2].
    for i in range(data.ndim):
        tmp_u = data
        for j in range(data.ndim):
            if j == i:
                continue
            tmp_u = n_mode_product(tmp_u, U[j].T, j)
        U[i] = np.linalg.svd(
            tensor_unfold(tmp_u, i),
            full_matrices=False
        )[0][:, :R[i]]
    # Get @G. Then we can construct the approximate data.
    G = n_mode_product(tmp_u, U[-1].T, data.ndim-1)
    approx_data = n_mode_product(G, U[0], 0)
    for i in range(1, data.ndim):
        approx_data = n_mode_product(approx_data, U[i], i)
    # @err: The error between the origin data and the approximate data.
    err = data - approx_data
    print('\t%d\t%lf' % (iters, np.linalg.norm(err)))
    # Break if it converges.
    if np.allclose(approx_data, prev_approx_data, 1e-5, 1e-6):
        break
    prev_approx_data = approx_data
t[1] += time.time() - t_start

print("Time performance:")
print(t[0], '/', t[1])
