import cupy as cp
import time

N = 20000

start = time.time()

x = cp.random.random((N, N), dtype=cp.float32)

y = cp.sin(x)

cp.cuda.Stream.null.synchronize()

end = time.time()

print(f"Shape: {y.shape}")
print(f"Time: {end-start:.2f} seconds")
