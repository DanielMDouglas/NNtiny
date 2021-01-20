from network import NN

# now, to show that it works
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import fmin_l_bfgs_b as minimizer

def loss(w, net, A, B): # define the two populations
    net.update_weights(w)
    
    Atrue = 1
    Btrue = 0

    Aloss = sum(pow(Atrue - net.output(Ai), 2)
                for Ai in A)/float(len(A))
    Bloss = sum(pow(Btrue - net.output(Bi), 2)
                for Bi in B)/float(len(B))
    return Aloss + Bloss

def train(net, A, B): # train against two data sets
    print (minimizer(loss,
                     [-25, -25, -25, -25, 2, 2],
                     args = (net, A, B),
                     approx_grad = True))

# Define two sets distributed with some separation in the input variable space
nA = 1000
A = st.multivariate_normal.rvs((-1, -1),
                               size = nA,
                               cov = 0.5)

nB = 1000
B = st.multivariate_normal.rvs((1, 1),
                               size = nB,
                               cov = 0.5)

def make_scatter_plot():
    plt.figure()
    plt.scatter(*A.T, color = 'r', label = 'A')
    plt.scatter(*B.T, color = 'b', label = 'B')
    plt.legend()
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'$x_1$')

def make_score_hist(title):
    Ascores = np.array([NN.output(Ai) for Ai in A])
    Bscores = np.array([NN.output(Bi) for Bi in B])

    plt.figure()
    bins = np.linspace(0, 1, 20)
    plt.hist(Ascores,
             bins = bins,
             color = 'r',
             label = 'A',
             histtype = 'step')
    plt.hist(Bscores,
             bins = bins,
             color = 'b',
             label = 'B',
             histtype = 'step')
    plt.title(title)
    plt.xlabel("score")
    plt.legend()

make_scatter_plot()
make_score_hist("Before Training")
    
train(NN, A, B)

make_score_hist("After Training")

plt.show()
