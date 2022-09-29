import numpy as np
w = 2
h = 3


x, y = np.meshgrid(
                np.arange(w, dtype = np.float32) +.5,
                np.arange(h, dtype = np.float32) + .5,
                indexing = "xy"
            )

print(x.shape, y.shape)

x = [[1,1],
     [2,2],
     [3,3]]
x = np.array(x)

print(x)
c = np.concatenate([x, x[-2:-1,:]],0)
print(c)
print(c.shape)
print(x.tolist())