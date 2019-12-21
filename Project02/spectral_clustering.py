from argparse import ArgumentParser
import scipy.io as sio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys


# Visualization.
#
# For problem 01: Show the points.
# For problem 02: Show the points with color according to different cluster.
class Visualizor:
    def __init__(self, title):
        _, ax = plt.subplots()
        ax.set_title(title)
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        self.ax = ax
    

    def plot(self, x, y, size=20, color='blue'):
        self.ax.scatter(x, y, s=size, c=color)


    def show(self):
        plt.show()

    
    def save(self, name):
        plt.savefig(name+'.png')


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
# @node_num: The number of nodes.
dist = sio.loadmat(args.pth)['data']
node_num = dist.shape[0]

##############################################################################
# Problem 01.
#
# Generate a position matrix @pos with the centroid of @pos's is origin (0,0).
# Plot a 2-d scatter plot. 
#
# @pos: Position matrix. @pos[i] = x_i
# @G: Defined as -0.5*(@dist - 1*D1^T - D1*1^T), where D1 is the first column
#   of @dist.
##############################################################################
# Let the position matrix as @pos. Define @G = @pos^T * @pos.
# Assume that x1 = 0, we get @G = -0.5(@dist - 1*D1^T - D1*1^T).
# Please note that G is a symmetric matrix.
vec1 = np.ones((node_num, 1))
G = -0.5 * (dist \
    - np.dot(vec1, dist[:, 0, None].T) \
    - np.dot(dist[:, 0, None], vec1.T))
# Eigenvalue decomposition: G = QAQ^T
eig_val, eig_vec = np.linalg.eigh(G)
pos = np.sqrt(eig_val[-2:]) * eig_vec[:, -2:]   # @pos = X = Q * sqrt(A)
print('x1 =', pos[0])
print('We can varify that when we get G by -0.5(D - 1*D1^T - D1*1^T), x1' +\
    ' will always be 0.')
pos -= np.mean(pos, axis=0) # Shift to the center.

# Visualization.
vis = Visualizor('Problem 01')
vis.plot(pos[:, 0], pos[:, 1], 20, 'blue')
vis.show()

##############################################################################
# Problem 02.
#
# Seperate the points into 2/3/4 classes by spectral clustering.
#
# @W: Edge weight matrix.
#   w_ij = 1 - (dist[i, j]-min(dist))/(max(dist)-min(dist)).
# @D: Node weight matrix. @D = diag(@W * 1), where 1 is a 40 x 40 matrix.
#   But in my implementation, I only calculate the diagonal part. I calculate
#   @W * 1 with 1 as a 40 x 1 vector, then put the result on the diagonal of
#   of a zero matrix.
# @L: Normalized weight Laplacian.
##############################################################################
# Setup node/edge weight matrix.
min_dist = np.min(dist+np.diag(np.full(node_num, np.inf)))
max_dist = np.max(dist)
W = 1 - (dist-min_dist) / (max_dist-min_dist)
diag_D = np.matmul(W, vec1).T[0]  # @diag_D: Diagonal of @D.
D = np.diag(diag_D)

# Spectral clustering.
# 
# I use normalized weight Laplacian.
# 
# @invh_D: D^-0.5. 'invh' stands for 'inverse' and 'half'.
# @L: Normalized graph Laplacian matrix. @L = I - @invh_D*W*@invh_D.
invh_D = np.diag(np.sqrt(1./diag_D))
L = np.identity(node_num) - np.dot(np.dot(invh_D, W), invh_D)
eig_val, eig_vec = np.linalg.eigh(L)

# K classes classifier. Clustering by Kmeans.
color = ['blue', 'red', 'green', 'cyan']
for K in range(2, 5):
    cls_vec = KMeans(
        n_clusters=K, random_state=0
        ).fit(eig_vec[:, 1:3]).labels_
    # Visualization.
    vis = Visualizor('Problem 02 - '+str(K)+' classes')
    for l in range(K):
        vis.plot(pos[cls_vec==l, 0], pos[cls_vec==l, 1], 20, color[l])
    vis.show()
