import numpy as np
import matplotlib.pyplot as plt

width = 1920
height = 1080

x = np.linspace(-2, 2, width)
y = np.linspace(-2, 2, height)

X, Y = np.meshgrid(x, y)

Z = X + 1j * Y

C = complex(-0.7, 0.27015)

iterations = np.zeros(Z.shape)

for i in range(100):
    mask = np.abs(Z) < 2
    Z[mask] = Z[mask]**2 + C
    iterations[mask] = i

plt.imshow(iterations, cmap="magma")
plt.axis("off")
plt.savefig("test.png", dpi=300)
