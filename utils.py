import numpy as np
import struct as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import sklearn.preprocessing as skp
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import logging
from SVM import Svm
import time
import sys

rng = np.random.default_rng()


# grid must have shape (parameters x values)
def grid_search(grid, x_train, y_train, kernel, tol=10e-3, eps=10e-3, b=1, passes=1000):
    res = []
    if kernel != 'radial':
        for i, gr in enumerate(grid):
            print(f"Training: #{i}")
            svm = Svm(gr, k_type=kernel, b=b, max_iters=passes, tol=tol, eps=eps)
            cost, (train_acc, test_acc), t_elapsed = k_fold_validation(svm, x_train, y_train, 5)
            print(f"Model: {kernel}, C: {gr}")
            print(
                f"Cost: {cost.mean():.3f}, Accuracies: {train_acc.mean():.3f}, {test_acc.mean():.3f}, Time: {t_elapsed.mean():.3f}")
            res.append((cost, test_acc, t_elapsed))
    else:
        grid = np.atleast_2d(grid)
        for i, gr in enumerate(grid.T):
            print(f"Training: #{i}")
            svm = Svm(gr[0], k_type=kernel, b=b, max_iters=passes, g=gr[1], tol=tol, eps=eps)
            cost, (train_acc, test_acc), t_elapsed = k_fold_validation(svm, x_train, y_train, 5)
            print(f"Model: {kernel}, C: {gr[0]}, gamma: {gr[1]}")
            print(
                f"Cost: {cost.mean():.3f}, Accuracies: {train_acc.mean():.3f}, {test_acc.mean():.3f}, Time: {t_elapsed.mean():.3f}")
            res.append((cost, test_acc, t_elapsed))
    return res


def k_fold_validation(model, x: np.ndarray, y: np.ndarray, k: int):
    cost = np.zeros(k)
    test_accs = np.zeros(k)
    train_accs = np.zeros(k)
    ts_elapsed = np.zeros(k)
    t_elapsed = 0.
    perm = rng.permutation(x.shape[0])
    # split data into k groups
    groups = np.array_split(perm, k)
    for i in range(k):
        print(f"Fold: {i}, last fold t: {t_elapsed:.3f}", end="\r")
        # use k-1 groups for training
        ind = np.concatenate(groups[:i] + groups[-1:i:-1][::-1])
        # fit data to model
        train_acc, t_elapsed = model.fit(x[ind].T, y[ind])
        # use k-th group to test model
        _l, _v, test_acc = model.predict(x[groups[i]].T, y[groups[i]])
        # collect stats
        cost[i] = model.cost()
        test_accs[i] = test_acc
        train_accs[i] = train_acc
        ts_elapsed[i] = t_elapsed
        print("                                                  ", end="\r")
    return cost, (train_accs, test_accs), ts_elapsed


def validate(model, x, y, x_test, y_test, passes, name="Unknown"):
    time_elapsed = np.zeros(passes)
    train_accs = np.zeros(passes)
    test_accs = np.zeros(passes)
    for i in range(passes):
        print('Training #', i, end="\r")
        train_acc, t_elapsed = model.fit(x.T, y)
        _l, _v, test_acc = model.predict(x_test.T, y_test)
        time_elapsed[i] = t_elapsed
        train_accs[i] = train_acc
        test_accs[i] = test_acc
    print('Model: ', name, ' Passes: ', passes)
    print(f"Training Acc: {train_accs.mean():.3f},Testing Acc: {test_accs.mean():.3f} Time: {time_elapsed.mean():.3f}")
    print("SV size: ", model.sv.size)
    return train_accs, test_accs, time_elapsed


def load(filename):
    file = open(filename, 'rb')
    file.seek(0)
    magic_number = st.unpack('>4B', file.read(4))
    dims = np.zeros((1, magic_number[3]), dtype=np.int32)
    for i in range(dims.size):
        dims[0][i] = st.unpack('>I', file.read(4))[0]
    total = np.prod(dims)
    data = np.asarray(st.unpack('>' + 'B' * total, file.read(total.item())))
    dims_tuple = tuple(map(tuple, dims))[0]
    return data, dims_tuple


def process_data(pca, one_fit=True):
    # extract binary files
    filename = {'train_images': 'Data/fashion/train-images-idx3-ubyte',
                'train_labels': 'Data/fashion/train-labels-idx1-ubyte',
                'test_images': 'Data/fashion/t10k-images-idx3-ubyte',
                'test_labels': 'Data/fashion/t10k-labels-idx1-ubyte'}
    x_train, trd_dims = load(filename['train_images'])
    y_train, trl_dims = load(filename['train_labels'])
    x_test, ted_dims = load(filename['test_images'])
    y_test, tel_dims = load(filename['test_labels'])

    x_train = x_train.reshape((trd_dims[0], trd_dims[1] * trd_dims[2]))
    x_test = x_test.reshape((ted_dims[0], trd_dims[1] * trd_dims[2]))

    # %%
    if pca > 0:
        pipe = Pipeline([('scaler', skp.MinMaxScaler()), ('pca', PCA(pca))])

        # scale and reduce data
        x_train = pipe.fit_transform(x_train)
        x_test = pipe.transform(x_test)

    # one-hot encode labels
    hot = ''  # labeling variable
    if one_fit:
        ohe = skp.OneHotEncoder(sparse=False)
        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))
        y_train = ohe.fit_transform(y_train)
        y_test = ohe.transform(y_test)
        hot = '_hot'
    np.save('Processed/x_train_{}.npy'.format(pca), x_train)
    np.save('Processed/x_test_{}.npy'.format(pca), x_test)
    np.save('Processed/y_train{}.npy'.format(hot), y_train)
    np.save('Processed/y_test{}.npy'.format(hot), y_test)


# plots correct and wrong predictions from a confusion matrix
def plot_acc(conf_matrix: np.ndarray, name=None):
    if name is None:
        name = 'plot_w_' + np.random.randint(0, 1000000)
    correct_pred = conf_matrix.diagonal()
    wrong_pred = conf_matrix.sum(axis=0) - correct_pred

    fig, ax = plt.subplots()
    ax.bar(range(10), correct_pred, label='Correct Predictions')
    ax.bar(range(10), wrong_pred, bottom=correct_pred, label='Wrong Predictions')

    ax.set_axisbelow(True)
    ax.grid(linestyle='--')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Samples')
    ax.set_xticks(range(10))
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.tight_layout()

    plt.savefig('./Plots/' + name + '.svg', dpi=300, format='svg')
    plt.show()


# plots wrong predictions of each class from confusion matrix
def plot_wrong(conf_matrix: np.ndarray, name=None):
    if name is None:
        name = 'plot_w_' + np.random.randint(0, 1000000)
    np.fill_diagonal(conf_matrix, 0)
    csum_conf = conf_matrix.cumsum(axis=0)
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    x_pos = range(len(classes))
    fig, ax = plt.subplots()

    pal = sns.color_palette('muted')
    ax.bar(range(10), conf_matrix[0, :], color=pal[0], label=classes[0])
    for i in range(1, 10):
        ax.bar(range(10), conf_matrix[i, :], color=pal[i], label=classes[i], bottom=csum_conf[i - 1, :])

    ax.set_axisbelow(True)
    ax.grid(linestyle='--')
    ax.set_ylabel('Samples classified as other class')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes)
    plt.xticks(rotation=-50)

    ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.tight_layout()

    plt.savefig('./Plots/' + name + '.svg', dpi=300, format='svg')
    plt.show()


def plot_cost(cost_list, step, epochs, batch, name):
    length = len(cost_list)
    cost = [None] * length
    train = [None] * length
    test = [None] * length
    for i in range(length):
        cost[i] = cost_list[i][0]
        train[i] = cost_list[i][1]
        test[i] = cost_list[i][2]

    x_ticks = np.arange(start=0, stop=len(cost) + 1, step=step)
    title = 'model: {}, step: {}, epochs: {}, batch: {}'.format(name, step, epochs, batch)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_title(title)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cost', color=color)
    ax1.set_xlim([0, epochs])
    ax1.set_ylim([0, 0.7])

    ax1.grid()
    ax1.plot(cost, color=color)
    fig.tight_layout()

    plt.savefig(f"./Plots/c_{name}_s{step}_e{epochs}_b{batch}.svg", dpi=300, format='svg')
    plt.show()
    fig2, ax2 = plt.subplots()

    ax2.set_title(title)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy %', color='k')
    ax2.set_xlim([0, epochs])
    ax2.set_ylim([70, 100])

    ax2.plot(train, label='Train Set')
    ax2.plot(test, label='Test Set')
    ax2.legend(loc='best')
    ax2.grid()
    fig2.tight_layout()
    plt.savefig(f"./Plots/a_{name}_s{step}_e{epochs}_b{batch}.svg", dpi=300, format='svg')

    plt.show()
