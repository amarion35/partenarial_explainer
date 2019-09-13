import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import time

# Rony, J., Hafemann, L.G., Oliveira, L.S., Ayed, I.B., Sabourin, R., Granger, E., 2018. 
# Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses. 
# arXiv:1811.09600 [cs].

class DDN:
    def __init__(self, model, loss, alpha=0.08, gamma=1):
        self.model = model
        self.loss = loss
        self.alpha = alpha
        self.gamma = gamma
        self.times = {
            'fit': 0,
            'grad': 0
        }

    def fit(self, x, target, n_iter, verbose=1):
        if verbose>0:
            def log_print(*args, **kwargs):
                print(*args, **kwargs)
        else:
            log_print = lambda s: None
        t = time.time()
        x = torch.tensor([x])
        target = torch.tensor([target], dtype=torch.long)
        delta = 0
        y = [x]
        epsilon = 1
        m = -1
        out = self._predict(y[-1])
        log_print('y={}, target={}, m={}, out={}'.format(y[-1], target.data.numpy(), m, out.data.numpy()))
        for k in range(n_iter):
            log_print('Epoch {}'.format(k), end='')
            g = m*self._grad(y[-1], out, target)
            g = self.alpha*(g/(g.norm()+0.0001))
            delta = delta+g
            if self._vote(out)==target:
                log_print(' - IN ', end='')
                epsilon = (1-self.gamma)*epsilon
            else:
                log_print(' - OUT', end='')
                epsilon = (1+self.gamma)*epsilon
            l = self.loss(out, target).data.numpy()
            y.append(x+epsilon*(delta/delta.norm()))
            out = self._predict(y[-1])
            log_print(' - distance: {}'.format((y[-1]-x).norm().data.numpy()))
        self.times['fit'] += time.time()-t
        return y

    def _predict(self, x):
        return self.model(x, training=False)

    def _grad(self, y, out, target):
        t = time.time()
        y = Variable(y, requires_grad=True)
        out = self._predict(y)
        loss = self.loss(out, target)
        loss.backward()
        g = y.grad
        assert bool(torch.isnan(g).any())==False
        self.times['grad'] += time.time()-t
        return g

    def _vote(self, out):
        return torch.argmax(out, dim=1)[0]