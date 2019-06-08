import numpy as np

a1 = np.array([[[1,2,3],[2,3,4]],
               [[3,4,5],[4,5,6]],
               [[5,6,7],[6,7,8]]])

print('a1',a1)
print(a1.shape,'\n')

a2 = np.zeros([3,1,3],int)

print('a2',a2)
print(a2.shape,'\n')

a3 = np.concatenate((a1,a2), axis = 1)

print('a3',a3)
print(a3.shape)