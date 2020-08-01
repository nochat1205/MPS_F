import numpy as np
import tensorly as tl
from math import sqrt


def list_inversion(tuple: tuple):
    """
    for unfold2, Retouch mode
    """
    Li = list(tuple)
    for i in range(len(Li)):
        for j in range(i):
            if Li[j] > Li[i]:
                Li[i] += 1
    return Li


def unfold2(tensor, mode=(0, 1)):
    """
    tensor mode-() unfold = tl.unfold 's pro
    """
    dim0 = 1
    mode = list_inversion(mode[::-1])
    for i in mode:
        dim0 *= tensor.shape[i]
        tensor = tl.moveaxis(tensor, i, 0)
    tensor = tl.reshape(tensor, (dim0, -1))
    return tensor


def random_MPS(d, D: int):
    """
    d: list or int
        list will build a mps, use list
        int will build a mps, len(mps) = d, shape is D, d, D
    build a romdom MPS
    """
    try:
        iter(d)
    except Exception:
        a = [i for i in range(d)]
    else:
        a = list(d)
    length = len(a)
    A = []
    for i, j in zip(range(length), a):
        if i == 0:
            A.append(np.random.randn(1, j, D))
        elif i == length - 1:
            A.append(np.random.randn(D, j, 1))
        else:
            A.append(np.random.randn(D, j, D))
    return A


def low_MPSnew(MPS0, sval_nums: int):
    """
    compression MPS to sval_nums, left to right
    """
    mps = MPS0[:]
    for i in range(len(mps)-1):
        C = np.tensordot(unfold2(mps[i]), tl.unfold(mps[i+1], 0), 1)
        U, Sigma, VT = np.linalg.svd(C, 0)
        temp = mps[i].shape[0]
        D = min(sval_nums, C.shape[0], C.shape[1])
        mps[i] = tl.fold(tl.moveaxis(U[:, :D], 1, 0), 2, (temp, -1, D))
        SVT = np.tensordot(np.diag(Sigma[0:D]), VT[:D], 1)

        mps[i+1] = np.reshape(SVT, (D, -1, mps[i+1].shape[2]))
    return mps


def dot_MPS(MPS):
    """
    Shrink MPS to a big tensor
    """
    B = np.eye(MPS[0].shape[0])
    for i in MPS:
        B = np.tensordot(B, i, 1)
    return B


def MPO_times_MPS(MPO, MPS):
    """
    use MPO to modify MPS
    """
    MPS0 = []
    for i, j in zip(MPO, MPS):
        MPS0.append(np.tensordot(MPO, MPS, 1))
    return MPS0


def overlap_2MPS(MPS1, MPS2):
    """
    Shrink two MPS
    """
    lenth = len(MPS1) + 2
    return np.tensordot(dot_MPS(MPS1), dot_MPS(MPS2), lenth)


def error_2MPS(MPS1, MPS2):
    """
    the error F
    """
    A = sqrt(overlap_2MPS(MPS1, MPS1))
    B = sqrt(overlap_2MPS(MPS2, MPS2))
    AB = abs(overlap_2MPS(MPS1, MPS2))
    return (A * A - 2 * AB + B * B)


def LR_canonical(mps, n):
    """
    canonical from left to right
    """
    for i in range(n):
        C = tl.unfold(mps[i], 2)
        C = tl.moveaxis(C, 1, 0)
        U, Sigma, VT = np.linalg.svd(C, 0)
        U2 = tl.moveaxis(U, 1, 0)
        mps[i] = tl.fold(U2, 2, (mps[i].shape[0], mps[i].shape[1], -1))
        SVT = np.tensordot(np.diag(Sigma), VT, 1)

        mps[i+1] = np.tensordot(SVT, mps[i+1], 1)
    return mps


def RL_canonical(mps, n):
    """
    canonical from right to LEFT
    """
    N = len(mps)
    for i in range(N-1, n-1, -1):
        C = tl.unfold(mps[i], 0)
        U, Sigma, VT = np.linalg.svd(C, 0)
        mps[i] = tl.fold(VT, 0, (-1, mps[i].shape[1], mps[i].shape[2]))
        US = np.tensordot(U, np.diag(Sigma), 1)

        mps[i-1] = np.tensordot(mps[i-1], US, 1)
    return mps


def Mix_canonical(mps, n):
    """
    canonical_mix from sides to i
    """
    return LR_canonical(RL_canonical(mps, n+1), n)


def w_canoniacal(B, A, i):
    """
    count wi for Iterative optimization A
    """
    lenB = len(B)
    IAB = dot_MPS(B)
    if i != 0:
        IAtemp = np.moveaxis(dot_MPS(A[:i]), i+1, 0)
        IAB = np.tensordot(IAtemp, IAB, i+1)
    if i != len(B)-1:
        IAtemp = np.moveaxis(dot_MPS(A[i+1:]), 0, lenB-i)
        IAB = np.tensordot(IAB, IAtemp, lenB-i)
    return IAB
