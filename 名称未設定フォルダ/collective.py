import numpy as np

t_max = 100
cell = 24

Q = np.zeros((t_max+1,cell+1),np.int8)

Q[0,:-1] = [1,0,1,1,1, 0,0,0,1,0, 1,0,1,1,1, 0,0,0,1,0, 1,1,0,0]
Q[0,-1] = Q[0,0]

for i in range(3):
    idx_move = np.logical_and(Q[i,:-1]==1,Q[i,1:]==0)
    Q[i+1,:-1] = np.where(idx_move,0,Q[i,:-1])
    Q[i+1,1:]= np.where(idx_move,1,Q[i+1,1:])
    Q[i+1,0] = 1 if idx_move[-1] else Q[i + 1,0]
    Q[i + 1,-1] = Q[i+1,0]
    print(i+1,Q[i,:-1])
    