# coding=utf-8
# tt 分解
from Findex import a_color_re
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
import imageio
import MPS
from math import sqrt




img_eg = imageio.mimread("img/cat.gif")
tensor = np.array(img_eg)
tensor = tensor[5:30, :, :, :3]
print(tensor.shape)

B = MPS.tensorToMPS(tensor)

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
