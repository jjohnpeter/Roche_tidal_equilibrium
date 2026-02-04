import matplotlib.pyplot as plt
import numpy as np

qstar = np.array([0.322, 0.52156, 0.81508, 1.16273, 1.63848, 2.17684, 2.98134, 4.41219, 6.46543, 9.03968, 11.46731])
Qbh = 1
Rstar = np.array([0.40466, 0.459306, 0.513519, 0.556985, 0.600267, 0.6366, 0.673930, 0.718619, 0.759217, 0.792126, 0.813101])
Reff = np.array([0.42953, 0.486305, 0.542041, 0.588059, 0.633149, 0.6713, 0.713268, 0.764914, 0.813883, 0.855181, 0.883322])
K = np.array([0.11766, 0.15684, 0.20349, 0.24846, 0.3, 0.35, 0.41147, 0.5, 0.6, 0.7, 0.77783])
#plt.scatter(K, Rstar/Reff, color = 'black', label = 'Entropy Dependent Roche Error')
plt.scatter(Qbh/qstar, Rstar, color = 'blue', label='Polytropic Roche Limit')
plt.scatter(Qbh/qstar, Reff, color = 'orange', label='Eggleton Roche Limit')
plt.xlabel('Mass Ratio')
plt.ylabel('Roche Radius [Rstar]')
plt.title('Eggleton vs Polytropic Roche Limit')

plt.legend()
plt.show()
