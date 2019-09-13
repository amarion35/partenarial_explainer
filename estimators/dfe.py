import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from functools import partial
import utils.nested_list_utils
import utils.tree_utils as tree_utils
from estimators.dfe_model import DFE_model, BigModel
import xgboost as xgb

relu = torch.nn.functional.relu

class DFE():

    def __init__(self, model, n_class, n_features, alpha, device=None):
        if device is None: device = torch.device('cpu')
        self.device = device

        if isinstance(model, xgb.core.Booster):
            model = tree_utils.model2table(model)
        self.model = model
        self.n_tree = len(model)
        self.n_features = n_features
        self.n_class = n_class

        self.paths = self._list_paths(self.model)
        self.A, self.V = self._buildAV(self.paths)
        self.F, self.L = self._buildFL(self.paths)

        self.W = self._init_W(init=alpha)

        self.A = self._set_Tensor_List(self.A)
        self.V = self._set_Tensor_List(self.V)
        self.F = self._set_Tensor_List(self.F)
        self.W = self._set_Tensor_List(self.W)

        self.A = self._sparsify(self.A)

        self.D = None
        self.D2 = None

        self.dfe_model = []
        for c in range(self.n_class):
            self.dfe_model.append(DFE_model(A=self.A[c], V=self.V[c], F=self.F[c], W=self.W[c]))

    def get_model(self):
        return BigModel(self.dfe_model)

    def _list_paths(self, model):
        paths = []
        for i, tree in enumerate(model):
            paths.append(tree_utils.list_paths(tree, node=None, parents=[]))
        return paths

    def _set_Tensor_List(self, A):
        for i, a in enumerate(A):
            A[i] = Variable(torch.FloatTensor(a)).to(self.device)
        return A

    def _set_Tensor(self, A):
        A = Variable(torch.FloatTensor(A)).to(self.device)
        return A

    def _sparsify(self, T):
        for i in range(len(T)):
            T[i] = T[i].to_sparse()
        return T

    def fit(self, X, Y, X_val=None, Y_val=None):
        S = tree_utils.predict_model(self.model, X, self.n_class)

        X = self._set_Tensor(X)
        Y = self._set_Tensor(Y)
        S = self._set_Tensor(S)
        
        if (not X_val is None) and (not Y_val is None):
            S_val = tree_utils.predict_model(self.model, X_val, self.n_class)
            X_val = self._set_Tensor(X_val)
            Y_val = self._set_Tensor(Y_val)
            S_val = self._set_Tensor(S_val)

    def _init_W(self, init):
        W = []
        for i in range(self.n_class):
            w = init * np.ones(shape=(self.A[i].shape[0], 2*self.n_features))
            W.append(w)
        return W

    # Predict manually by browsing the trees
    def predict_model(self, model, X):
        pred = []
        if isinstance(model, xgb.core.Booster):
            model = tree_utils.model2table(model)
        pred = [self._predict_tree(t, x) for x in X for t in model]
        pred = np.reshape(pred, (len(X), len(model)))
        res = [[np.sum(sample[i::self.n_class]) for i in range(self.n_class)] for sample in pred]
        return res

    def _predict_tree(self, tree, x):
        res = float(self._browse_tree(tree, x, node=None, path=[])[-1]['leaf'])
        return res

    def _browse_tree(self, tree, x, node=None, path=[]):
        path = []
        if node is None:
            node = tree[0]
        while not node['is_leaf']:
            path.append(node)
            if x[int(node['split'])] <= node['split_condition']:
                node = tree[int(node['yes'])]
            else:
                node = tree[int(node['no'])]
        path.append(node)
        return path

    def _buildAV(self, paths):
        n_cond = 2*self.n_features
        A = []
        V = []

        for c in range(self.n_class):
            n_leaf = len([0 for i, tree_path in enumerate(paths) for path in tree_path if i%self.n_class==c])
            A_class = np.zeros(shape=(n_leaf, n_cond))
            V_class = np.zeros(shape=(n_leaf, n_cond))
            l = 0
            for t in range(self.n_tree//self.n_class):
                tree_paths = paths[c+(t*self.n_class)]
                for p in range(len(tree_paths)):
                    path = tree_paths[p]
                    if len(path)<2:
                        continue
                    for _, node in enumerate(path[:-1]):
                        feature = int(node['split'])
                        v = float(node['split_condition'])
                        if int(node['condition'])==1:
                            V_class[l, 2*feature] = min(V_class[l, 2*feature],v) if A_class[l, 2*feature]==1 else v
                            A_class[l, 2*feature] = 1
                        elif int(node['condition'])==-1:
                            V_class[l, (2*feature)+1] = min(V_class[l, (2*feature)+1],-v) if A_class[l, (2*feature)+1]==-1 else -v
                            A_class[l, (2*feature)+1] = -1
                        else:
                            raise Exception("Condition verification is undefined")
                    l += 1
            A.append(A_class)
            V.append(V_class)
        return A, V

    def _buildFL(self, paths):
        F = []
        L = []
        for c in range(self.n_class):
            f = []
            l = []
            for t in range(len(paths)//self.n_class):
                for path in paths[c+(t*self.n_class)]:
                    f.append(path[-1]['leaf'])
                    l.append(path[-1]['nodeid'])
            F.append(np.array(f))
            L.append(np.array(l))
        return F, L

    def _predict(self, X):
        out = []
        for c in range(self.n_class):
            out.append(self.dfe_model[c].forward(X))
        out = torch.stack(out).to(self.device)
        out = out.transpose(0,1)
        return out

    def predict(self, X, softmax=True):
        X = self._set_Tensor(X)
        H = self._predict(X)
        if not softmax:
            return H.cpu().data.numpy()
        return F.softmax(H).cpu().data.numpy()