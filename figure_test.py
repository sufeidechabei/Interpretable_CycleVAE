import matplotlib.pyplot as plt
import numpy as np
y = []
for x in range(512):
    y.append(np.random.randint(0, 512))
data = np.arange(512) + 1
plt.bar(data, y, facecolor = 'g')
plt.grid(True, linestyle = 'dashed')
plt.show()
a = 'filter'
index = []
for x in range(512):
    index.append(a+str(x + 1))
plt.xticks(data, index)
plt.savefig('./figure.jpg')
plt.show()
