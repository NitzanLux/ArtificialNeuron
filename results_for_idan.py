import matplotlib.pyplot as plt
import numpy as np
#%%
data = [(1, 0.9690, "original"), (1, 0.9781, "reduction"),
        (3, 0.9859, "original"), (5, 0.9866, "original"),
        (7, 0.9869, "original"), (7, 0.9872, "reduction"),
        (5, 0.9882, "reduction"), (3, 0.9885, "reduction")]
data = sorted(data,key=lambda x:2*x[0]+int(x[2]=="original"))
o_data=list(map(lambda x:x[1],data[::2]))
plt.plot(o_data)
plt.show()