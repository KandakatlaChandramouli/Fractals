import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

WIDTH = 3840
HEIGHT = 2160

MAX_ITER = 500

C = complex(-0.7, 0.27015)

xmin, xmax = -1.8, 1.8
ymin, ymax = -1.2, 1.2

start = time.time()

x = cp.linspace(xmin, xmax, WIDTH)
y = cp.linspace(ymin, ymax, HEIGHT)

X, Y = cp.meshgrid(x, y)

Z = X + 1j * Y

output = cp.zeros(Z.shape, dtype=cp.uint16)

mask = cp.ones(Z.shape, dtype=cp.bool_)

for i in range(MAX_ITER):

    Z[mask] = Z[mask] * Z[mask] + C

    escaped = cp.abs(Z) > 2

    newly_escaped = escaped & mask

    output[newly_escaped] = i

    mask &= ~escaped

    if not cp.any(mask):
        break

output = cp.asnumpy(output)

plt.figure(figsize=(16,9))
plt.imshow(output, cmap="magma")
plt.axis("off")
plt.savefig("outputs/images/julia_4k.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0)

end = time.time()

print(f"Render Time: {end-start:.2f} seconds")
print("Saved outputs/images/julia_4k.png")
