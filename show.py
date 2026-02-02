import numpy as np
import matplotlib.pyplot as plt
xs = np.arange(100)
ys = 200* xs - 10000
plt.plot(xs, ys)
a = np.random.rand()
b = np.random.rand()
c = np.random.rand() * 0.5
x = np.random.rand(50) + 50
y = 200* x - 10000 + 10*x**2
x = x + np.random.randn(50) *0.5
plt.scatter(x, y, color='red')
# Fit line
p = a * xs + b
px = a * x + b + c * x**2
loss = np.mean((px - y) ** 2)
for i in range(1000000):
    dp = np.mean(2 * (px - y) * x)
    db = np.mean(2 * (px - y))
    dc = np.mean(2 * (px - y) * x**2)
    c -= 0.00001 * dc
    a -= 0.0001 * dp
    b -= 0.1 * db
    p = a * xs + b
    px = a * x + b
    loss = np.mean((px - y) ** 2)
    print(f'Loss: {loss}')
plt.scatter(x, px, color='green')
plt.plot(xs, p)
plt.xlim(0, 100)
plt.ylim(0, 200)
plt.show()