# forked version of Yubei Chen, Sparse Manifold Transform Lib Ver 0.1
"""
This file contains multiple method to sparsify the coefficients
"""
import numpy as np
import torch

def quadraticBasisUpdate(basis, Res, ahat, lowestActivation, HessianDiag, stepSize = 0.001,constraint = 'L2', Noneg = False):
    """
    This matrix update the basis function based on the Hessian matrix of the activation.
    It's very similar to Newton method. But since the Hessian matrix of the activation function is often ill-conditioned, we takes the pseudo inverse.

    Note: currently, we can just use the inverse of the activation energy.
    A better idea for this method should be caculating the local Lipschitz constant for each of the basis.
    The stepSize should be smaller than 1.0 * min(activation) to be stable.
    """
    dBasis = stepSize*(Res @ ahat.t())/ahat.size(1)
    dBasis = dBasis.div_(HessianDiag+lowestActivation)
    basis = basis.add_(dBasis)
    if Noneg:
        basis = basis.clamp(min = 0.)
    if constraint == 'L2':
        basis = basis.div_(basis.norm(2,0))
    return basis

def ISTA_PN(I,basis,lambd,num_iter,eta=None):
    # This is a positive-negative PyTorch-Ver ISTA solver
    # MAGMA uses CPU-GPU hybrid method to solve SVD problems, which is great for single task. When running multiple jobs, this flag should be turned off to leave the svd computation on only GPU.
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        #L = torch.max(torch.symeig(basis @ basis.t(),eigenvectors=False)[0])
        L = torch.linalg.eigvalsh(basis @ basis.t())[-1] # elements are in ascending order
        eta = 1./L

    #Res = torch.zeros(I.size()).type(dtype)
    #ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.FloatTensor(I.size()).fill_(0).to(I.device)
    ahat = torch.FloatTensor(M,batch_size).fill_(0).to(I.device)

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t() @ Res)
        ahat_sign = torch.sign(ahat)
        ahat.abs_()
        ahat.sub_(eta * lambd).clamp_(min = 0.)
        ahat.mul_(ahat_sign)
        Res = I - basis @ ahat
    return ahat, Res

def FISTA(I,basis,lambd,num_iter,eta=None):
    # This is a positive-only PyTorch-Ver FISTA solver
    #if device is not None: # this is a workaround to avoid cusolver error, see https://github.com/pytorch/pytorch/issues/60892#issue-931902763
    #    torch.cuda.set_device(device)
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        L = torch.linalg.eigvalsh(basis @ basis.t())[-1] # torch.max(torch.symeig(basis @ basis.t(),eigenvectors=False)[0])
        eta = 1./L

    tk_n = 1.
    tk = 1.
    Res = torch.FloatTensor(I.size()).fill_(0).to(I.device)
    ahat = torch.FloatTensor(M,batch_size).fill_(0).to(I.device)
    ahat_y = torch.FloatTensor(M,batch_size).fill_(0).to(I.device)

    for t in range(num_iter):
        tk = tk_n
        tk_n = (1+np.sqrt(1+4*tk**2))/2
        ahat_pre = ahat
        Res = I - (basis @ ahat_y)
        ahat_y = ahat_y.add(eta * basis.t() @ Res)
        ahat = ahat_y.sub(eta * lambd).clamp(min = 0.)
        ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk-1)/(tk_n)))
    Res = I - (basis @ ahat)
    return ahat, Res

def ISTA(I,basis,lambd,num_iter,eta=None):
    # This is a positive-only PyTorch-Ver ISTA solver
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        L = torch.linalg.eigvalsh(basis @ basis.t())[-1] # torch.max(torch.symeig(basis @ basis.t(),eigenvectors=False)[0])
        eta = 1./L

    #Res = torch.zeros(I.size()).type(dtype)
    #ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.FloatTensor(I.size()).fill_(0).to(I.device)
    ahat = torch.FloatTensor(M,batch_size).fill_(0).to(I.device)

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t() @ Res)
        ahat = ahat.sub(eta * lambd).clamp(min = 0.)
        Res = I - basis @ ahat
    return ahat, Res
