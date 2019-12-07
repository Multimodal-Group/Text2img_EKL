import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import os
import pickle
from PIL import Image


def imgs2ndarray(img_root, save=False):
    if save and os.path.exists(img_root+'.np.pkl'):
        imgs= pickle.load(open(img_root+'.np.pkl', 'rb'))
        print('>>>>>>>>use saved imgs.np.pkl')
    else:
        img_names= os.listdir(img_root)
        imgs= []
        for img in img_names:
            img= Image.open(img_root+'/'+img)
            img.load()
            img= np.asarray(img, dtype='int32')
            imgs.append(img)
        if save:
            pickle.dump(imgs, open(img_root+'.np.pkl', 'wb'))
    return imgs

class SpectralNorm(nn.Module):
    """Spectral normalization of weight with power iteration
    """
    def __init__(self, module, niter=1): # module eg. conv/linear
        super().__init__()
        self.module = module
        self.niter = niter
        self.init_params(module)

    @staticmethod
    def init_params(module):
        """u, v, W_sn
        """
        w = module.weight
        height = w.size(0)
        width = w.view(w.size(0), -1).shape[-1] # rule both 2d/3d

        u = nn.Parameter(torch.randn(height, 1), requires_grad=False)
        v = nn.Parameter(torch.randn(1, width), requires_grad=False)
        module.register_buffer('u', u)
        module.register_buffer('v', v)

    @staticmethod
    def update_params(module, niter):
        u, v, w = module.u, module.v, module.weight
        height = w.size(0)

        for i in range(niter):  # Power iteration
            v = w.view(height, -1).t() @ u
            v /= (v.norm(p=2) + 1e-12)
            u = w.view(height, -1) @ v
            u /= (u.norm(p=2) + 1e-12)

        w.data /= (u.t() @ w.view(height, -1) @ v).data  # Spectral normalization

    def forward(self, x):
        self.update_params(self.module, self.niter)
        return self.module(x)


class CondInstanceNorm(nn.Module):
    '''Cond BN'''
    def __init__(self, num_features, num_labels, eps=1e-5, momentum=0.1):
        super(CondInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(torch.Tensor(num_labels, num_features))
        self.bias = Parameter(torch.Tensor(num_labels, num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input, label):
        self._check_input_dim(input)
        b, c = input.size(0), input.size(1)

        # Repeat stored stats and affine transform params
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        input_reshaped = input.contiguous().view(1, b * c, *input.size()[2:])

        weight_per_sample = F.embedding(label.long(), self.weight).reshape(b * c)
        bias_per_sample = F.embedding(label.long(), self.bias).reshape(b * c)

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight_per_sample, bias_per_sample,
            True, self.momentum, self.eps)

        # Reshape back
        self.running_mean.copy_(running_mean.view(b, c).mean(0, keepdim=False))
        self.running_var.copy_(running_var.view(b, c).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])

    def eval(self):
        return self

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                .format(name=self.__class__.__name__, **self.__dict__))

class CondInstanceNorm1d(CondInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

class CondInstanceNorm2d(CondInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class CondInstanceNorm3d(CondInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))


def dir_sampling(labels, alpha= (0.05,)*10):
    '''sampling from special dirichlet distribution for adding noise for one-hot
    '''
    ls= []
    for lb in labels:
        while True:
            s= np.random.dirichlet(alpha, 1)[0]
            if s[lb]< 0.8:
                continue
            ls.append(s)
            break
    return np.array(ls)

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1, 1, 1))).to(device).float()
    # Get random linear interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates=Variable(interpolates,requires_grad=True)
    d_interpolates = D(interpolates)[-1] # for two output of D
    grad_weight = Variable(torch.ones(d_interpolates.size()), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=grad_weight, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_gradient_penalty_withcond(D, cls, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1, 1, 1))).to(device).float()
    # Get random linear interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates=Variable(interpolates,requires_grad=True)
    d_interpolates = D(interpolates, cls)[-1] # for two output of D
    grad_weight = Variable(torch.ones(d_interpolates.size()), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=grad_weight, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty