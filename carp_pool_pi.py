from itertools import repeat
import multiprocessing as mp
import random
from timeit import timeit


def calc_pi(N):
    M = 0
    for i in range(N):
        # Simulate impact coordinates
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # True if impact happens inside the circle
        if x**2 + y**2 < 1.0:
            M += 1
    return (4 * M / N, N)  # result, iterations


def submit(ctx, N):
    with ctx.Pool() as pool:
        pool.starmap(calc_pi, repeat((N,), 4))


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    for i in (1_000, 100_000, 10_000_000):
        res = timeit(lambda: submit(ctx, i), number=5)
        print(i, res)

