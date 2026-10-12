import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- OPERAÇÕES ALGÉBRICAS ---- #
def uniform_cross_dissolve(f1, f2, t):
    ff1 = f1.astype(np.float64)
    ff2 = f2.astype(np.float64)
    return np.clip((1 - t) * ff1 + t * ff2, 0, 255).astype(np.uint8)

def non_uniform_cross_dissolve(f1, f2):
    R, C = f1.shape
    i, j = np.ogrid[:R, :C]
    t = (i + j) / (R + C - 2)
    return np.clip((1 - t) * f1.astype(np.float64) + t * f2.astype(np.float64), 0, 255).astype(np.uint8)
    
# ---- TRANSFORMAÇÕES DE INTENSIDADE ---- #
def negative(f):
    return 255 - f

def log(f):
    ff = f.astype(np.float64)
    N = ff.max()
    if N == 0: return f.copy()
    c = 255 / np.log(1 + N)
    return np.clip(c * np.log(1 + ff), 0, 255).astype(np.uint8)

def gamma(f, p):
    ff = f.astype(np.float64)
    N = ff.max()
    if N == 0: return f.copy()
    c = 255 / np.power(N, p)
    return np.clip(c * np.power(ff, p), 0, 255).astype(np.uint8)

def contrast_stretch(f):
    ff = f.astype(np.float64)
    r = ff / 255.0
    s = np.zeros_like(r)
    
    mask1 = (r >= 0) & (r < 3/8)
    s[mask1] = ((1/8) / (3/8)) * r[mask1]
    
    mask2 = (r >= 3/8) & (r < 5/8)
    s[mask2] = ((7/8 - 1/8) / (5/8 - 3/8)) * (r[mask2] - 3/8) + 1/8
    
    mask3 = (r >= 5/8) & (r <= 1)
    s[mask3] = ((1 - 7/8) / (1 - 5/8)) * (r[mask3] - 5/8) + 7/8
    
    return np.clip(s * 255.0, 0, 255).astype(np.uint8)

def threshold(f, t=127):
    f_res = f.copy()
    f_res[f_res > t] = 255
    f_res[f_res <= t] = 0
    return f_res

# ---- HISTOGRAMA ---- #
def histogram_equalization(f):
    h = np.zeros(256, dtype=np.int32)
    for i in range(256): h[i] = len(f[f == i])
    height, width = f.shape
    p = h / (height * width)
    
    T = np.zeros(256, dtype=np.float64)
    acumulado = 0.0
    for i in range(256):
        acumulado += p[i]
        T[i] = 255.0 * acumulado
    return np.round(T).astype(np.uint8)[f]

def histogram_expansion(f):
    ff = f.astype(np.float64)
    f_min, f_max = ff.min(), ff.max()
    if f_max == f_min: return f.copy()
    return np.clip(np.round((ff - f_min) / (f_max - f_min) * 255.0), 0, 255).astype(np.uint8)

# ---- CONTRASTE ADAPTATIVO ---- #
def adaptive_contrast(f, n, c):
    if n % 2 == 0:
        raise ValueError("O tamanho da vizinhança n deve ser ímpar!")
    ff = f.astype(np.float64)
    height, width = ff.shape
    radius = n // 2
    ff_pad = np.pad(ff, pad_width=radius, mode="edge")
    g = np.zeros_like(ff)
    
    for i in range(height):
        for j in range(width):
            nh = ff_pad[i : i + n, j : j + n]
            mu, sigma = np.mean(nh), np.std(nh)
            if sigma != 0:
                g[i, j] = mu + (c / sigma) * (ff[i, j] - mu)
            else:
                g[i, j] = ff[i, j]
    return np.clip(g, 0, 255).astype(np.uint8)

# ---- FILTRAGEM ESPACIAL ---- #
def convolution(f, w, offset=0, mode="constant"):
    height, width = f.shape
    m, n = w.shape
    
    # Validação de kernel ímpar
    if m % 2 == 0 or n % 2 == 0:
        raise ValueError("As dimensões do kernel (m x n) devem ser ímpares!")
        
    a, b = (m - 1) // 2, (n - 1) // 2
    
    f_float = f.astype(np.float64)
    f_padded = np.pad(f_float, pad_width=((a, a), (b, b)), mode=mode)
    g = np.zeros_like(f_float)
    
    for x in range(height):
        for y in range(width):
            k = 0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    k += w[a + s, b + t] * f_padded[a + x - s, b + y - t]
            g[x, y] = k + offset
    return np.clip(g, 0, 255).astype(np.uint8)

def mean(f, m_size, n_size):
    if m_size % 2 == 0 or n_size % 2 == 0:
        raise ValueError("As dimensões do filtro da média devem ser ímpares!")
    w = np.ones((m_size, n_size), dtype=np.float64) / (m_size * n_size)
    return convolution(f, w, mode="edge")

def median(f, m_size, n_size):
    if m_size % 2 == 0 or n_size % 2 == 0:
        raise ValueError("As dimensões do filtro da mediana devem ser ímpares!")
    height, width = f.shape
    a, b = (m_size - 1) // 2, (n_size - 1) // 2
    f_padded = np.pad(f.astype(np.float64), pad_width=((a, a), (b, b)), mode="constant")
    g = np.zeros_like(f, dtype=np.float64)
    
    for x in range(height):
        for y in range(width):
            window = f_padded[x:(x + m_size), y:(y + n_size)]
            g[x, y] = np.median(window.ravel())
    return np.clip(g, 0, 255).astype(np.uint8)

# ---- AGUÇAMENTO ---- #
def laplacian_sharpening(f):
    w = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    g = convolution(f, w)
    return np.clip(f.astype(np.float64) + g, 0, 255).astype(np.uint8)

def highboost_sharpening(f, k=1.0):
    f_float = f.astype(np.float64)
    w = np.ones((3, 3), dtype=np.float64) / 9.0
    f_blur = convolution(f_float, w, mode="edge")
    g_mask = f_float - f_blur.astype(np.float64)
    return np.clip(f_float + (k * g_mask), 0, 255).astype(np.uint8)

# ---- DETECÇÃO DE BORDAS ---- #
def roberts(f):
    w1 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
    w2 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    g1 = convolution(f, w1, mode="edge").astype(np.float64)
    g2 = convolution(f, w2, mode="edge").astype(np.float64)
    magnitude = np.abs(g1) + np.abs(g2)
    return np.clip(np.abs(g1), 0, 255).astype(np.uint8), \
           np.clip(np.abs(g2), 0, 255).astype(np.uint8), \
           np.clip(magnitude, 0, 255).astype(np.uint8)

def sobel(f):
    Gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx = convolution(f, Gx_kernel, mode="edge").astype(np.float64)
    Gy = convolution(f, Gy_kernel, mode="edge").astype(np.float64)
    magnitude = np.sqrt(np.square(Gx) + np.square(Gy))
    return np.clip(np.abs(Gx), 0, 255).astype(np.uint8), \
           np.clip(np.abs(Gy), 0, 255).astype(np.uint8), \
           np.clip(magnitude, 0, 255).astype(np.uint8)

def laplacian_edges(f):
    w = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    g = convolution(f, w, mode="edge")
    return np.clip(np.abs(g), 0, 255).astype(np.uint8)

def kirsch(f):
    masks = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    ]
    responses = [convolution(f, w, mode="edge").astype(np.float64) for w in masks]
    magnitude = np.max(np.stack(responses, axis=0), axis=0)
    return np.clip(magnitude, 0, 255).astype(np.uint8)