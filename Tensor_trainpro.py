# coding=utf-8
# tt 分解
from Findex import a_color_re
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
import imageio
import MPS
from math import sqrt


# mode-(0,1)
# just success in (I1, I2, I3, I4...)mode-0 == (I1,I2, I3I4I5...)mode-0
# mode-1,2 可用,->更泛用


img_eg = imageio.mimread("img/cat.gif")
tensor = np.array(img_eg)
tensor = tensor[5:30, :, :, :3]
print(tensor.shape)

# left to right
B = []
r = []
# 二维数组存储r

##############################################
M = tl.unfold(tensor, 0)
U, Sigma, VT = np.linalg.svd(M, 0)
r.append([Sigma.shape[0]])
B.append(U.reshape(1, U.shape[0], -1))


# 思路
# SVD = I1r1 r1I...
# r1 I2....->r1I2 I2i3..->r1I2r2 r2...
for i in range(tensor.ndim - 2):
    C = np.diag(Sigma).dot(VT)
    r[i].extend(tensor.shape[i+1:])
    M = tl.fold(C, 0, tuple(r[i]))
    M = MPS.unfold2(M)
    U, Sigma, VT = np.linalg.svd(M, 0)
    r.append([Sigma.shape[0]])
    B.append(U)


C = np.diag(Sigma).dot(VT)
B.append(C.reshape(C.shape[0], -1, 1))

######################################################
B[1] = tl.moveaxis(B[1], 1, 0) # 旋转轴
B[2] = tl.moveaxis(B[2], 1, 0)
B[1] = tl.fold(B[1], 2, (r[0][0], tensor.shape[1], r[1][0])) #->rnInrn+1
B[2] = tl.fold(B[2], 2, (r[1][0], tensor.shape[2], r[2][0]))

# 混合正则化
B = MPS.Mix_canonical(B, 1)
# B[0] = np.squeeze(B[0]) # 输出维数修饰
# B[3] = np.squeeze(B[3])


# 变分
sval_nums = 10
t = 5
A = MPS.low_MPSnew(B, sval_nums) # 暴力切片
bef = np.squeeze(MPS.dot_MPS(A))
y = []
for j in range(t):
    y.append(MPS.error_2MPS(A, B))
    for i in range(4):
        A = MPS.Mix_canonical(A, i)
        B = MPS.Mix_canonical(B, i)
        A[i] = MPS.w_canoniacal(B, A, i)


aft = np.squeeze(MPS.dot_MPS(A))
n = np.arange(t)
plt.plot(n, y)
plt.show()
# print(np.tensordot(A[0], B[0], [(1, 0), (1, 0)]))

fig, ax = plt.subplots(1, 3, figsize=(24, 32))

ax[0].imshow(tensor[3].astype(np.uint8))
ax[0].set(title="all")


ax[1].imshow(a_color_re(bef[3]).astype(np.uint8))
ax[1].set(title="all")

ax[2].imshow(a_color_re(aft[3]).astype(np.uint8))
ax[2].set(title="all")
plt.show()
