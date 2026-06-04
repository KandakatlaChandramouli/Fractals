import os
import gc
import time
import cupy as cp
import matplotlib.pyplot as plt

WIDTH = 3840
HEIGHT = 2160

MAX_ITER = 500

C = complex(-0.7, 0.27015)

FRAME_DIR = "/content/drive/MyDrive/FractalFrames"

os.makedirs(FRAME_DIR, exist_ok=True)

CENTER_X = 0.31
CENTER_Y = -0.08

BASE_WIDTH = 3.6
ASPECT_RATIO = HEIGHT / WIDTH
BASE_HEIGHT = BASE_WIDTH * ASPECT_RATIO

ZOOM_FACTOR = 0.985

TOTAL_FRAMES = 120

def render_frame(frame_idx):

    scale = ZOOM_FACTOR ** frame_idx

    current_width = BASE_WIDTH * scale
    current_height = BASE_HEIGHT * scale

    xmin = CENTER_X - current_width / 2
    xmax = CENTER_X + current_width / 2

    ymin = CENTER_Y - current_height / 2
    ymax = CENTER_Y + current_height / 2

    x = cp.linspace(xmin, xmax, WIDTH)
    y = cp.linspace(ymin, ymax, HEIGHT)

    X, Y = cp.meshgrid(x, y)

    Z = X + 1j * Y

    output = cp.zeros(Z.shape, dtype=cp.float32)

    mask = cp.ones(Z.shape, dtype=cp.bool_)

    for i in range(MAX_ITER):

        Z[mask] = Z[mask] * Z[mask] + C

        escaped = cp.abs(Z) > 2

        newly_escaped = escaped & mask

        if cp.any(newly_escaped):

            smooth = (
                i
                + 1
                - cp.log(
                    cp.log(
                        cp.abs(Z[newly_escaped])
                    )
                )
                / cp.log(2.0)
            )

            output[newly_escaped] = smooth

        mask &= ~escaped

        if not cp.any(mask):
            break

    img = cp.asnumpy(output)

    filename = os.path.join(
        FRAME_DIR,
        f"frame_{frame_idx+1:03d}.png"
    )

    plt.figure(
        figsize=(3840/300,2160/300),
        dpi=300
    )

    plt.imshow(
        img,
        cmap="magma",
        origin="lower"
    )

    plt.axis("off")

    plt.savefig(
        filename,
        dpi=300,
        pad_inches=0
    )

    plt.close("all")

    del img
    del output
    del mask
    del Z

    cp.get_default_memory_pool().free_all_blocks()

    gc.collect()

    zoom_percent = (1.0 / scale - 1.0) * 100

    print(
        f"Frame {frame_idx+1:03d}/{TOTAL_FRAMES}"
        f" | Zoom {zoom_percent:.2f}%"
    )

if __name__ == "__main__":

    start = time.time()

    for frame in range(TOTAL_FRAMES):
        render_frame(frame)

    end = time.time()

    print()
    print("================================")
    print("RENDER COMPLETE")
    print("================================")
    print(f"Total Time: {end-start:.2f} sec")
