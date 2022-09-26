import numpy as np
w = 2
h = 3


x, y = np.meshgrid(
                np.arange(w, dtype = np.float32) +.5,
                np.arange(h, dtype = np.float32) + .5,
                indexing = "xy"
            )


print(np.stack([x,y, np.ones_like(x)]))
