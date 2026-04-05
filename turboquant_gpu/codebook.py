# lloyd-max optimal scalar quantizer
# rotation makes each coordinate ~ N(0, 1/d), so we solve 1d k-means
# against that gaussian. runs once on cpu at init.

import math
import torch
from scipy import integrate


def _gaussian_pdf(x, sigma):
    return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(
        -x * x / (2 * sigma * sigma))


def solve_lloyd_max(d, bits, max_iter=200, tol=1e-10):
    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)
    pdf = lambda x: _gaussian_pdf(x, sigma)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32))


class LloydMaxCodebook:

    def __init__(self, d, bits):
        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits
        self.centroids, self.boundaries = solve_lloyd_max(d, bits)

    def quantize(self, x):
        diffs = x.unsqueeze(-1) - self.centroids.to(x.device)
        return diffs.abs().argmin(dim=-1)

    def dequantize(self, indices):
        return self.centroids.to(indices.device)[indices.long()]

    def __repr__(self):
        return f"LloydMaxCodebook(d={self.d}, bits={self.bits}, levels={self.n_levels})"
