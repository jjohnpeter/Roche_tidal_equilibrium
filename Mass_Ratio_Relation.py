import matplotlib.pyplot as plt
import numpy as np

qstar = np.array([0.32200, 1.16273, 1.63848, 2.17684, 11.46731, 0.52156, 0.81508, 2.98134, 4.41219, 6.46543, 9.03968])
Qbh = 1
RocheErr = np.array([0.94209, 0.94716, 0.948066, 0.9483, 0.920504, 0.944481, 0.947380, 0.94484, 0.939477, 0.932833, 0.926267])

plt.scatter(qstar/Qbh, RocheErr)
plt.xlabel('Mass Ratio')
plt.ylabel('Roche Ratio')
plt.title('Mass Ratio Dependent Roche Limit Error')

plt.show()
