import numpy as np

# 클래스 표.
# C0: 인스타,  C1: 잠,  C2: 객프,  C3: 소개방,  C4: 동역학,  C5: 골치,  C6: 출튀 
Reward = np.array([-10., 0., -2., -2., -2., 1., 10.])
Reward = np.transpose(Reward)

PMatrix = np.zeros((7,7)) # 전이 확률 행렬
PMatrix[0, 0], PMatrix[1, 1], PMatrix[2, 2], PMatrix[3, 3], PMatrix[4, 4], PMatrix[5, 5], PMatrix[6, 6] = 0.9, 0., 0., 0., 0., 0., 0.
PMatrix[0, 2] = 0.1
PMatrix[2, 0], PMatrix[2, 3] = 0.5, 0.5
PMatrix[3, 1], PMatrix[3, 4] = 0.2, 0.8
PMatrix[4, 5], PMatrix[4, 6] = 0.4, 0.6
PMatrix[5, 2], PMatrix[5, 3], PMatrix[5, 4] = 0.2, 0.4, 0.4
PMatrix[6, 1] = 1.

VMatrix = np.zeros(7)
VMatrix = np.transpose(VMatrix)
VMatrix_pre = np.zeros(7)

Identity = np.eye(7)
D_F = 0.5
K = 1000

print("State transisition Probability Matrix: ")
print (PMatrix, "\n")
print("Value function: ")
print(VMatrix, "\n")
print("Reward :")
print(Reward, "\n")

for k in range(1, K):
    VMatrix_pre = VMatrix
    PVMatrix = np.matmul(PMatrix, VMatrix)
    print(PVMatrix.shape)
    VMatrix = Reward + D_F * PVMatrix
    norm = np.sum(abs(VMatrix - VMatrix_pre))
    print (k)
    if (norm < 0.00001):
        break

print(VMatrix)

PVInverse = np.linalg.inv(Identity - D_F * PMatrix)
print (np.matmul(PVInverse, Reward))