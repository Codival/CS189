# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt

import pydot

eps = 1e-5  # a small number
np.random.seed(7)

class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None, m=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.m = m
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    def information_gain(self,X, y, thresh):
        # TODO implement information gain function
        indices_r = np.argwhere(X>=thresh)
        indices_l = np.argwhere(X<thresh)
        flat_r = [val for sublist in indices_r for val in sublist]
        flat_l = [val for sublist in indices_l for val in sublist]
        pred_y_r = [y[index] for index in flat_r]
        pred_y_l = [y[index] for index in flat_l]
        total_impurity = self.gini_impurity(X,y,thresh)
        right_impurity = (len(pred_y_r))*self.gini_impurity(X,pred_y_r,thresh)
        left_impurity = (len(pred_y_l))*self.gini_impurity(X,pred_y_l,thresh)
        return total_impurity - (right_impurity + left_impurity)/len(y)

    def gini_impurity(self,X, y, thresh):
        # TODO implement gini_impurity function
        if len(y)==0:
            return 0
        prob = np.count_nonzero(y)/len(y)
        return 2*(prob*(1-prob))
        

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        idx = range(X.shape[1])
        if self.max_depth > 0:
            if self.m != None:
                idx = np.random.choice(X.shape[1],self.m,replace=False)
                #feat_X = X[:,idx]
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features,m=self.m)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features,m=self.m)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat


class BaggedTrees:
    def __init__(self, params=None, n=200, max_depth=3, feature_labels=None, m=None):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTree(max_depth, feature_labels,m)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        for dt in self.decision_trees:
            idx = np.random.choice(X.shape[0],X.shape[0],replace=True)
            bootx = X[idx,:]
            dt.fit(bootx,y[idx])
        return self

    def predict(self, X):
        # TODO implement function
        predictions = []
        for dt in self.decision_trees:
            predictions += [dt.predict(X)]
#         print(np.asmatrix(predictions).shape)
#         for pred in predictions:
#             print(pred[:100])
        return stats.mode(predictions)[0][0]
        


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, max_depth=3, feature_labels=None, m=1):
        if params is None:
            params = {}
        # TODO implement function
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTree(max_depth, feature_labels,m)
            for i in range(self.n)
        ]


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO implement function
        return self

    def predict(self, X):
        # TODO implement function
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)


if __name__ == "__main__":
    dataset = "titanic"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)

    
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X,y,test_size=.2, random_state=7)
    
    
    print("\n\nPart 0: constant classifier")
    print("Training:", 1 - np.sum(y) / y.size)
    
    
    # Basic decision tree
    print("\n\nDecision tree:")
    if dataset == "titanic":
        depth = 6
    else:
        depth = 16
    dt = DecisionTree(max_depth=depth, feature_labels=features)
    dt.fit(X_train, y_train)
    train_pred = np.count_nonzero(np.equal(dt.predict(X_train),y_train))/len(y_train)
    val_pred = np.count_nonzero(np.equal(dt.predict(X_val),y_val))/len(y_val)
    print("Training:", train_pred)
    print("Validation:", val_pred)
    pred = dt.predict(Z)
    
    # Basic Random Forest
    print("\n\nRandom Forest:")
#     plot_x =[]
#     acc_train = []
#     acc_val = []
#     for i in range(1,100,10):
#         print(i)
#         plot_x += [i]
    rf = RandomForest(n=20, max_depth=5, feature_labels=features, m=5)
    rf.fit(X_train, y_train)
    train_pred = np.count_nonzero(np.equal(rf.predict(X_train),y_train))/len(y_train)
    val_pred = np.count_nonzero(np.equal(rf.predict(X_val),y_val))/len(y_val)
#         acc_train += [train_pred]
#         acc_val += [val_pred]
#     print('prediction:', rf.predict(X_train)[:100])
    print("Training:", train_pred)
    print("Validation", val_pred)
#     plt.plot(plot_x,acc_train)
#     plt.plot(plot_x,acc_val)
#     plt.show()
#     pred = rf.predict(Z)
    
    
    print("\n\nPart (c): sklearn's decision tree")
    clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    sklearn.tree.export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    graph = pydot.graph_from_dot_data(out.getvalue())
    pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # TODO implement and evaluate parts c-h
