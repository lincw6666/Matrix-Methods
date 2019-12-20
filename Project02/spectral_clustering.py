from argparse import ArgumentParser
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


class Visualizor:
    def __init__(self, title):
        _, ax = plt.subplots()
        ax.set_title(title)
        # ax.grid(True, which='both')
        ax.set_xlim(-1., 1.)
        ax.set_ylim(-6.e-4, 4.e-4)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        self.ax = ax
    

    def plot(self, x, y, size=20, color='blue'):
        self.ax.scatter(x, y, s=size, c=color)


    def show(self):
        plt.show()


# Load arguments.
parser = ArgumentParser()
parser.add_argument(
    '--path',
    help='The path of distance.mat.',
    dest='pth'
)

args = parser.parse_args()
if args.pth is None:
    print('Error!! Expect 1 arguments: `--path`!!\n')
    parser.print_help()
    sys.exit(0)

# Store data to a ndarray.
# @dist: The distance matrix. dist.shape = 40 x 40
dist = sio.loadmat(args.pth)['data']

##############################################################################
# Problem 01.
#
# Generate a position matrix @pos with the centroid of @pos's is origin (0,0).
# Plot a 2-d scatter plot. 
#
# @pos: Position matrix.
# @G: Defined as -0.5*(@dist - 1*D1^T - D1*1^T), where D1 is the first column
#   of @dist.
##############################################################################
# Let the position matrix as @pos. Define @G = @pos^T * @pos.
# Assume that x1 = 0, we get @G = -0.5(@dist - 1*D1^T - D1*1^T).
# Please note that G is a symmetric matrix.
vec1 = np.ones(dist.shape[0])
G = -0.5 * (dist - np.dot(vec1, dist[:, 1].T) - np.dot(dist[:, 1], vec1.T))
eig_val, eig_vec = np.linalg.eigh(G)
pos = eig_vec[:, -2:]
pos -= np.mean(pos, axis=0)

# Visualization.
vis = Visualizor('Problem 01')
vis.plot(pos[:, 0], pos[:, 1], 20, 'blue')
vis.show()

##############################################################################
# Problem 02.
#
# Seperate the points into 2/3/4 classes by spectral clustering.
#
# @W: Edge weight matrix. w_ij = 1 - (dist[i, j]-min(dist))/(max(dist)-min(dist)).
# @D: Node weight matrix. @D = @W^T * @W.
# @L: Normalized weight Laplacian.
##############################################################################
# Setup node/edge weight matrix.
min_dist = np.min(dist+np.diag(np.full(dist.shape[0], np.inf)))
max_dist = np.max(dist)
W = 1 - (dist-min_dist) / (max_dist-min_dist)
# np.fill_diagonal(W, 0.)
diag_D = np.diag(np.dot(W.T, W))    # @diag_D: Diagonal of D.
D = np.diag(diag_D)

# Spectral clustering.
# 
# I use normalized weight Laplacian.
# 
# @invh_D: D^-0.5. 'invh' stands for 'inverse' and 'half'.
invh_D = np.diag(np.sqrt(1./diag_D))
L = np.identity(dist.shape[0]) - np.dot(np.dot(invh_D, W), invh_D)
eig_val, eig_vec = np.linalg.eigh(L)
# Build classify vector @cls_vec.
cls_vec = eig_vec[:, 1]

# Visualization.
vis = Visualizor('Problem 02 - 2 classes')
vis.plot(pos[cls_vec<0., 0], pos[cls_vec<0, 1], 20, 'blue')
vis.plot(pos[cls_vec>=0., 0], pos[cls_vec>=0, 1], 20, 'red')
vis.show()
