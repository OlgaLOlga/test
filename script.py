import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

os.chdir('D:/lab1/py')
os.getcwd()

# 1 matrica i massiv
a = 10
a_rand = np.random.randint(10)
a_zero = 0
a_ones = 1

b = np.array([1, 2, 3])
b_rand = np.random.randint(10, size=5)
b_zero = np.zeros(5)
b_ones = np.ones(5)

c = np.array([[1, 2, 3], [4, 5, 6]])
c_rand = np.random.randint(10, size=(2, 3))
c_zero = np.zeros((2, 3))
c_ones = np.ones((2, 3))

# 2 zagruzka dannyh
data = np.loadtxt('data.txt')

# 3 zagruzka dannyh, ocenka parametrov
data = scipy.io.loadmat('data/1D/var1.mat')
print (data.keys())
n = data['n']

max_    = np.max(n)     
min_    = np.min(n)     
mean_   = np.mean(n)    
median_ = np.median(n) 
var_    = np.var(n)     
std_    = np.std(n)     

# 4  Graph odnomer velichiny i plotnost`
plt.plot(n, 'k')
plt.hlines(mean_, 0, len(n), colors='r', linestyles='solid')
plt.hlines(mean_ + std_, 0, len(n), colors='g', linestyles='dashed')
plt.hlines(mean_ - std_, 0, len(n), colors='g', linestyles='dashed')
plt.grid()
plt.show()


plt.hist(n, bins=25)
plt.grid()
plt.show()

# 5 Avtocorelyac
def autocorrelate(a):
  n = len(a)
  cor = []
  for i in range(n//2, n//2+n):
    a1 = a[:i+1]   if i < n else a[i-n+1:]
    a2 = a[n-i-1:] if i < n else a[:2*n-i-1]
    cor.append(np.corrcoef(a1, a2)[0, 1])
  return np.array(cor)

n_1d = np.ravel(n)
cor = autocorrelate(n_1d)
plt.plot(cor)
plt.show()

# 6  Mnogomern dannye
data = scipy.io.loadmat('data/ND/var1.mat')
print (data.keys())
mn = data['mn']

# 7 matrica correlyacii
n = mn.shape[1]
corr_matrix = np.zeros((n, n))

for i in range(0, n):
  for j in range(0, n):
    corr_matrix[i, j] = np.corrcoef(mn[:, i], mn[:, j])[0, 1]

np.set_printoptions(precision=2)
print(corr_matrix)

plt.plot(mn[:, 2], mn[:, 5], 'b.')
plt.grid()
plt.show()
