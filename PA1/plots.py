import matplotlib.pyplot as plt
import numpy as np


a=np.array([6179, 6179, 6179, 6179, 6180, 5862, 6180, 6191, 6180, 6188, 6182, 6180, 6127, 5604, 5301, 5711, 4614, 5187, 4959, 5701, 5556, 6494, 6698, 8550, 7336, 6398, 6220, 6181, 6281, 6782, 6446, 8106, 7674, 8851, 6736, 8690, 7272, 8419, 8601, 6292, 6907, 8133, 8212, 7884, 8843, 6949, 8264, 6833, 6842, 9775, 8824])
a = a/55000.0
b=np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550])

#a=np.array([1135, 1135, 1135, 1135, 1134, 1065, 1136, 1135, 1135, 1136, 1133, 1135, 1116, 1024, 979, 1083, 842, 960, 915, 1090, 1009, 1203, 1258, 1604, 1379, 1168, 1143, 1135, 1159, 1257, 1190, 1546, 1443, 1639, 1271, 1645, 1381, 1613, 1634, 1151, 1300, 1539, 1567, 1500, 1688, 1310, 1564, 1288, 1295, 1873, 1663])
#a = a/10000.0





#plt.subplot(221)
#a=np.array([170,175,180,185,190,195,200,205])
#b=np.array([10.37,10.375,10.38,10.39,10.41,10.42,10.425,10.43])

#plt.ylabel('Test set Accuracy')
plt.ylabel('Training Accuracy')
plt.xlabel('Number of iterations')
plt.plot(b,a,'k',marker='o')
plt.title('Training accuracy over 2550 iterations for Sigmoid activatinons (0.001 - learning rate)')


'''plt.subplot(222)

x=np.array([130,135,140,145,150])
y=np.array([10.36,10.37,10.39,10.41,10.42])	
plt.ylabel('frequency')
plt.xlabel('Voltage')
plt.title('3rd Node')

plt.plot(x,y,'r')'''

plt.show()