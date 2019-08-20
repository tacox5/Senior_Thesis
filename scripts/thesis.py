import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.cos(x)
z = np.cos(x)

ccf = np.correlate(y, z, mode = 'same')
ccc = y*z-y.shape[0]*y.mean()*z.mean()/((y.shape[0]-1)*np.std(y)*np.std(z))

idx = np.argmax(np.abs(ccc))

#ccc /= ccc[idx]

plt.subplot(3,1,1)
plt.plot(x,ccf)

plt.subplot(3,1,2)
plt.plot(x,ccc)

plt.subplot(3,1,3)
plt.plot(x,y)
plt.plot(x,z)

plt.show()
