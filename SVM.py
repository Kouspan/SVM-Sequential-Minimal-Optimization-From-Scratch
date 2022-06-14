import time
from typing import Optional, Any
import numexpr as ne
import numpy as np



rng = np.random.default_rng()


# KERNELS


# args[0] is sa constant with value 1 by default

def linear(x1, x2, *args):
    return np.dot(x1.T, x2) + args[0]


def quad(x1, x2, *args):
    return (np.dot(x1.T, x2) + args[0])**2


def poly(x1, x2, *args):
    return (np.dot(x1.T, x2) + args[0])**args[1]


# https://stackoverflow.com/a/47271663/7473428
# fast radial kernel calculation
# args[0] is gamma constant, 0.5 by default
def radial(x1, x2, *args):
    x1_norm = np.sum(x1**2, axis=0)
    x2_norm = np.sum(x2**2, axis=0)
    k = ne.evaluate('exp(-g*(A+B-2*C))', {
        'A': x1_norm[:, None],
        'B': x2_norm[None,],
        'C': np.dot(x1.T, x2),
        'g': args[0]
    })
    return k


kernels = {'linear': linear, 'quad': quad, 'poly': poly, 'radial': radial}


class Svm:
    """Support Vector Machine.
        Has 3 different kernels, linear, polynomial and radial.
        Uses Sequential Minimal Optimization for classification. """

    def __init__(self, c, k_type, eps=10e-3, tol=10e-3, max_iter=1000, **kwargs):
        self.c = c  # C regularization parameter
        self.k_type = k_type  # Type of kernel
        self.eps = eps  # Threshold error
        self.tol = tol  # Tolerance of KKT conditions
        self.b = 0  # threshold
        self.max_iter = max_iter
        self.args = []  # Arguments for kernel
        self.a = None  # Lagrange multipliers
        self.y = None  # Labels
        self.x = None  # Training Data
        self.e = None  # Error Cache.
        self.sv = None  # Indices of support vectors
        self.iters = None

        # Initialize kernel arguments
        if k_type != 'radial':
            self.args.append(kwargs.get('b', 1))
            if k_type == 'poly':
                self.args.append(kwargs.get('t', 3))
        else:
            self.args.append(kwargs.get('g', 0.1))

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
            Trains the SVM with Sequential Minimal Optimization.
            Each iteration optimises a pair of Lagrange multipliers.
        :param x: np.ndarray, shape must be (Features x Samples)
        :param y: np.ndarray, labels must be -1 and 1
        :returns acc: training accuracy
        :returns elapsed: time elapsed
        """

        # initialize self variables
        samples = x.shape[1]
        self.x = x
        self.y = y
        self.a = np.zeros(samples)
        self.e = np.full(samples, np.nan)
        self.b = 0
        self.iters = 0
        # Outer loop, finds the 1st Lagrange multiplier (a2)
        # When the index of a1 is found, examine if the example is valid for optimization
        # The loop breaks if after the 1st pass, no non-bound alphas have been optimised and in
        #   the next pass none of all the alphas have changed

        num_changed = 0
        examine_all = True
        start = time.perf_counter()
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:  # pass from all the examples
                for i in range(samples):
                    num_changed += self.examine_example(i)  # examine if the example is valid for optimization
            else:  # pass only from non bound examples (a!= 0 & a!=C)
                ind = self.non_bound_a()
                for i in ind:
                    num_changed += self.examine_example(i)  # examine if the example is valid for optimization
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            if self.iters > self.max_iter:
                break
            self.iters += 1

        end = time.perf_counter()
        self.sv = np.flatnonzero(self.a)  # select indices of the support vectors (a != 0)
        _l, _v, acc = self.predict(self.x, self.y)
        elapsed = end - start
        return acc, elapsed

    def examine_example(self, i):
        """
        Examines if an example is valid for optimization and finds the 2nd Lagrange Multiplier (a1)
        :param i: index of alpha to be examined
        :return: 1 if alphas are changed and 0 otherwise
        """
        a2 = self.a[i]
        e2 = self.get_e(i)  # check in the error cache first, else calculate the error of ith example
        r2 = e2 * self.y[i]

        # check if the KKT conditions are satisfied within the tolerance
        if (r2 < -self.tol and a2 < self.c) or (r2 > self.tol and a2 > 0):
            ind = self.non_bound_a()
            # 1st second choice heuristic
            # find the a1 tha maximizes the step |e1-e2|
            if ind.size > 1:
                j = self.heuristic(i, ind)
                if self.take_step(i, j):
                    return 1
            # 2nd second choice heuristic
            # search all non-bound alphas starting from a random index
            for j in np.roll(ind, rng.integers(-1, ind.size, size=1)):
                if self.take_step(i, j):
                    return 1
            # 3rd second choice heuristic
            # search all alphas starting from a random index
            for j in np.roll(range(self.a.size), rng.integers(-1, self.a.size, size=1)):
                if self.take_step(i, j):
                    return 1
        return 0

    def take_step(self, i, j):
        """
        Optimizes the alpha pair.
        :param i:  2nd Lagrange multiplier a2
        :param j: 1st Lagrange multiplier a1
        :return: True if alphas were changed, False otherwise
        """
        if i == j:
            return False
        # initialize for ease of use later and updating
        b = self.b
        y1 = self.y[j]
        y2 = self.y[i]
        a1 = self.a[j]
        a2 = self.a[i]
        e1 = self.get_e(j)
        e2 = self.get_e(i)
        s = self.y[i] * self.y[j]

        # Find the bounds of a2 based on the constraints of QP
        low, high = self.bounds(i, j)

        # No optimization can be done, return False
        if low == high:
            return False

        k11 = self.kernel(j, j)
        k12 = self.kernel(j, i)
        k22 = self.kernel(i, i)

        # second derivative of the objective function
        eta = k11 + k22 - 2 * k12
        # update a2
        if eta > 0:
            self.a[i] = a2 + (y2 * (e1 - e2) / eta)
            # clip  a2 to be within bounds
            if self.a[i] < low:
                self.a[i] = low
            elif self.a[i] > high:
                self.a[i] = high
        else:
            # calculate bounds of objective function for a2 at bounds
            l_obj, h_obj = self.obj_bounds(i, j, low, high, k11, k22, k12)
            if l_obj < h_obj - self.eps:
                self.a[i] = low
            elif l_obj > h_obj + self.eps:
                self.a[i] = high

        # round alpha to bounds if their difference is smaller than 1e-8
        if self.a[i] < 1e-8:
            self.a[i] = 0
        elif self.a[i] > self.c - 1e-8:
            self.a[i] = self.c

        # if step is smaller than epsilon, don't make optimization
        if abs(self.a[i] - a2) < self.eps * (self.a[i] + a2 + self.eps):
            return False
        self.a[j] = a1 + s * (a2 - self.a[i])

        if self.a[j] < 1e-8:
            self.a[j] = 0
        elif self.a[j] > self.c - 1e-8:
            self.a[j] = self.c

        # calculate the threshold
        if 0 < self.a[j] < self.c:
            b1 = e1 + y1 * (self.a[j] - a1) * k11 + y2 * (self.a[i] - a2) * k12 + self.b
            self.b = b1
        elif 0 < self.a[i] < self.c:
            b2 = e2 + y1 * (self.a[j] - a1) * k12 + y2 * (self.a[i] - a2) * k22 + self.b
            self.b = b2
        else:
            b1 = e1 + y1 * (self.a[j] - a1) * k11 + y2 * (self.a[i] - a2) * k12 + self.b
            b2 = e2 + y1 * (self.a[j] - a1) * k12 + y2 * (self.a[i] - a2) * k22 + self.b
            self.b = (b1 + b2) / 2

        # update error cache for optimized alphas, if they are non-bound
        for t in [j, i]:
            if 0 < self.a[t] < self.c:
                self.e[t] = 0

        # update error cache for all non-bound alphas that weren't part of the optimization
        ind = self.non_bound_a([j, i])
        self.e[ind] = self.e[ind] + y1 * (self.a[j] - a1) * self.kernel(j, ind) + y2 * (self.a[i] - a2) * self.kernel(i,
            ind) + b - self.b

        return True

    def obj_bounds(self, i, j, low, high, k11, k22, k12):
        y1 = self.y[j]
        y2 = self.y[i]
        e1 = self.e[j]
        e2 = self.e[i]
        s = y1 * y2
        b = self.b
        a1 = self.a[j]
        a2 = self.a[i]

        f1 = y1 * (e1 + b) - a1 * k11 - s * a2 * k12
        f2 = y2 * (e2 + b) - s * a1 * k12 - a2 * k22
        l1 = a1 + s * (a2 - low)
        h1 = a1 + s * (a2 - high)
        obj_low = l1 * f1 + low * f2 + 0.5 * l1**2 * k11 + 0.5 * low**2 * k22 + s * low * l1 * k12
        obj_high = h1 * f1 + high * f2 + 0.5 * h1**2 * k11 + 0.5 * high**2 * k22 + s * high * h1 * k12
        return obj_low, obj_high

    # returns the index which maximizes |E1-E2| given E2
    def heuristic(self, i, ind):
        if self.e[i] >= 0:
            return np.argmin(self.e[ind])
        else:
            return np.argmax(self.e[ind])

    # return the indices for all non-zero, non-C alphas
    def non_bound_a(self, *args) -> np.ndarray:
        ind = np.logical_and(self.a != 0, self.a != self.c)  # select all non-bound alphas
        if len(args) > 0:
            for i in args[0]:
                ind[i] = False
        ind = np.flatnonzero(ind)  # return indices of non-bound alphas
        return ind

    # returns the error of an example
    def get_e(self, i):
        if np.isnan(self.e[i]):  # error hasn't been calculated yet
            self.e[i] = self.f(i) - self.y[i]  # calculate and return
        return self.e[i]

    # return
    def bounds(self, i, j):
        """
        Calculates the bounds of a2, given a1
        :param i: index of a2
        :param j: index of a1
        :return: bounds of a2
        """
        if self.y[i] * self.y[j] < 0:  # y1 != y2
            low = max(0, self.a[i] - self.a[j])
            high = min(self.c, self.c + self.a[i] - self.a[j])
        else:  # y1 == y2
            low = max(0, self.a[i] + self.a[j] - self.c)
            high = min(self.c, self.a[i] + self.a[j])
        return low, high

    def f(self, i):
        """
        Output function of SVM.
        for internal use only.
        Uses only the alphas of the support vectors.
        :param i: index of training example
        :return: output of svm
        """
        ind = np.flatnonzero(self.a)
        if ind.size == 0:  # all alphas are zero
            return self.b
        k = self.kernel(i, ind)
        return np.sum(self.a[ind] * self.y[ind] * k) - self.b

    def kernel(self, i, ind):
        ind = np.atleast_1d(ind)  # make sure ind is at least 1d, so self.x[:, ind] keeps its shape
        return kernels[self.k_type](self.x[:, [i]], self.x[:, ind], *self.args)

    def predict(self, x: np.ndarray, y=None) -> tuple[Any, np.ndarray, Optional[Any]]:
        """
        Predicts class of x.
        x can be an array of samples  or just one sample
        :param x: sample or array of samples
        :param y: labels of x
        :return:    vals: svm decision function of x
                    labels: class of x based on decision
                    acc: Accuracy if y is given, else None
        """
        acc = None
        k = kernels[self.k_type](x, self.x[:, self.sv], *self.args)
        vals = np.sum(self.a[self.sv] * self.y[self.sv] * k, axis=1) - self.b
        labels = np.ones(vals.shape)
        labels[vals <= 0] = -1
        if y is not None:
            correct = labels == y
            acc = correct.mean()

        return labels, vals, acc

    def cost(self):
        """Cost function of SVM"""
        # K(i,j) matrix
        k = kernels[self.k_type](self.x[:, self.sv], self.x[:, self.sv], *self.args)
        y = np.atleast_2d(self.y[self.sv])
        # Y(i,j) matrix
        y = np.dot(y.T, y)
        a = np.atleast_2d(self.a[self.sv])
        # A(i,j) matrix
        a = np.dot(a.T, a)
        # multiply Y,A,K and sum both axes
        cost = 0.5 * np.sum(y * a * k) - np.sum(a)
        return cost
