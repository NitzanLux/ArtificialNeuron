import matplotlib.pyplot as plt
import numpy as np
#%%
data = [(1, 0.9676, "original"), (1, 0.9751, "reduction"),
        (3, 0.9834, "original"), (3, 0.9873, "reduction"),
        (5,  0.9853, "original"), (5, 0.9878, "reduction"),
        (7, 0.9866, "original"), (7, 0.9870, "reduction")]
data = sorted(data,key=lambda x:2*x[0]+int(x[2]=="original"))
o_data=list(map(lambda x:x[1],data[1::2]))
r_data=list(map(lambda x:x[1],data[::2]))
p = plt.plot([1,3,5,7],o_data,label="original")
color = p[0].get_color()
plt.scatter([1,3,5,7],o_data,50,marker="*",c=color)
p = plt.plot([1,3,5,7],r_data,label="reduction")
color = p[0].get_color()
plt.scatter([1,3,5,7],r_data,50,marker="*",c=color)
plt.legend()
plt.xlabel("Number of layers")
plt.ylabel("AUC validation")
plt.title("Reduction comparison")
plt.show()