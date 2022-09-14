import EntropyHub as EH
import numpy as np
import matplotlib.pyplot as plt
MOBJ = EH.MSobject()
a=np.zeros((20,4))
a[:,0]=np.sin(np.arange(10,30)*np.pi/10)
a[:,1]=np.sin(np.arange(20)*np.pi/10)
a[:,2]=np.cos(np.arange(10,30)*np.pi/10)
a[:,3]=np.cos(np.arange(20)*np.pi/10)
plt.plot(a)
b = EH.XMSEn(a,MOBJ)
plt.show()
