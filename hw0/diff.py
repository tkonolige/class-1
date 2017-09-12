import numpy as np
from math import *
from matplotlib import pyplot

# computes the derivative using the forward difference formula
# returns len(x)-1 x len(x) matrix
def diffmat(x):
    n = len(x)
    return ((np.eye(n-1, n, k=1) - np.eye(n-1, n)).T / (x[1:] - x[:-1])).T

# backwards difference
def diffmat_back(x):
    n = len(x)+1
    return (((np.eye(n, n, k=-1) - np.eye(n, n))[1:,:]).T / x).T

# computes the second derivative
# returns len(x)-2 x len(x) matrix
def diff2mat(x):
    n = len(x)
    dists = x[:-2]-x[2:]
    return diffmat_back(dists/2).dot(diffmat(x))

def accuracy(points, func, difffunc, diff2func):
    d = diffmat(points).dot(np.vectorize(func)(points))
    d2 = diff2mat(points).dot(np.vectorize(func)(points))
    d_gt = np.vectorize(difffunc)(points)[1:]
    d2_gt = np.vectorize(diff2func)(points)
    # pyplot.plot(points,d2_gt)
    # pyplot.plot(points[1:-1],d2, 'o')
    # pyplot.show()
    n = np.linalg.norm(d - d_gt)/sqrt(len(points))
    n2 = np.linalg.norm(d2 - d2_gt[1:-1])/sqrt(len(points))
    return n,n2

def test_acc(space, name):
    d_acc = []
    d2_acc = []
    sizes = np.array([10,20,40,80,160])
    for i in sizes:
        ls = space(-1,1,i)
        # from the class notebook
        f = tanh
        diff = lambda x: cosh(x)**(-2)
        diff2 = lambda x: -2*tanh(x)*cosh(x)**(-2)
        n, n2 = accuracy(ls, f, diff, diff2)
        d_acc.append(n)
        d2_acc.append(n2)
    pyplot.loglog(sizes, d_acc, 'o', label='1st derivative')
    pyplot.loglog(sizes, d2_acc, 'o', label='2nd derivative')
    pyplot.loglog(sizes, (sizes-1)**(-1.), label='$h$')
    pyplot.loglog(sizes, (sizes-1)**(-2.), label='$h^2$')
    pyplot.title('Derivatives of tanh, %s spacing' % name)
    pyplot.legend()
    pyplot.show()

# scale a set of points so they fit on the given iterval
def scale(points, start, end):
    r = (end-start)/(points[-1] - points[0])
    pts = r * (points - points[0])
    return pts + start

test_acc(np.linspace, "linear")

def exp_points(start, end, num):
    return scale(np.exp(np.linspace(-4, 1, num)), start, end)

test_acc(exp_points, "exp")

def sin_endpoints(start, end, num):
    return scale(np.vectorize(lambda x: sin(x-pi/2)+1)(np.linspace(0, pi, num)), start, end)

test_acc(sin_endpoints, "sin")

def sin_centered(start, end, num):
    return scale(np.vectorize(lambda x: asin(x)+pi/2)(np.linspace(-1,1,num)), start, end)

test_acc(sin_centered, "asin")