import numpy as np
a = np.array([0,1,2,3,4,5])*2
# print a
# print a.ndim
# print a.shape
#
# b = a.reshape((3,2)).copy()
# print b
# print b.ndim
# print b.shape
# b[1][0] = 77
# print b

# print a*2
# print a**2

# print a[np.array([2,3,4,0])]
# print a[a > 2]

# Dados faltantes
c = np.array([1,2, np.NAN, 3, 4])
print c
print np.isnan(c)
c = c[~np.isnan(c)]
print c