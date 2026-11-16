"""Ручні реалізації класичних просторових фільтрів та текстурних операторів.

Усі алгоритми (падінг, 2D-згортка, Гаусова фільтрація, медіанна фільтрація,
оператор Собеля та локальні бінарні шаблони LBP) реалізовані без використання
готових функцій фільтрації зі сторонніх бібліотек комп'ютерного зору.
"""

import numpy as np
from src.utils import to_grayscale


SOBEL_KERNEL_X = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

SOBEL_KERNEL_Y = np.array(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ],
    dtype=np.float64,
)


def _validate_kernel_size(kernel_size):
    """Перевіряє, що розмір маски є непарним додатним цілим числом."""
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"Розмір ядра kernel_size має бути непарним числом >= 1, отримано: {kernel_size}"
        )


def pad_image(image, pad_h, pad_w, mode="edge"):
    """Додає відступи до двовимірного зображення заданим способом (edge/reflect/zero)."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Очікується непорожня двовимірна матриця для падінгу.")
    if pad_h == 0 and pad_w == 0:
        return arr.copy()

    if mode == "zero":
        return np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0.0)
    if mode == "edge":
        return np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    if mode == "reflect":
        # Для матриць 1x1 reflect у деяких режимах зводиться до edge
        if min(arr.shape) == 1:
            return np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
        return np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    raise ValueError(f"Непідтримуваний режим падінгу: {mode}")


def build_gaussian_kernel(kernel_size=3, sigma=1.0):
    """Формує нормоване 2D ядро Гауса за аналітичною формулою."""
    _validate_kernel_size(kernel_size)
    if sigma <= 0.0:
        raise ValueError("Параметр sigma для Гаусового ядра має бути > 0.")

    if kernel_size == 1:
        return np.array([[1.0]], dtype=np.float64)

    radius = kernel_size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis)
    kernel = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * (sigma ** 2)))
    return kernel / float(np.sum(kernel))


def manual_convolve2d(image, kernel, pad_mode="edge"):
    """Виконує ручну дискретну 2D-згортку матриці зображення з довільним ядром."""
    arr = to_grayscale(image).astype(np.float64)
    kernel_arr = np.asarray(kernel, dtype=np.float64)
    if kernel_arr.ndim != 2:
        raise ValueError("Ядро згортки повинно бути двовимірною матрицею.")

    k_h, k_w = kernel_arr.shape
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError("Розміри ядра згортки повинні бути непарними.")

    height, width = arr.shape
    padded = pad_image(arr, k_h // 2, k_w // 2, mode=pad_mode)
    result = np.zeros((height, width), dtype=np.float64)

    for ky in range(k_h):
        for kx in range(k_w):
            weight = float(kernel_arr[ky, kx])
            if weight != 0.0:
                result += weight * padded[ky : ky + height, kx : kx + width]

    return result


def manual_gaussian_filter(image, kernel_size=3, sigma=1.0, pad_mode="edge"):
    """Застосовує ручний Гаусів фільтр (розширення реалізації з CGV_5.py)."""
    kernel = build_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    filtered = manual_convolve2d(image, kernel, pad_mode=pad_mode)
    return np.clip(np.rint(filtered), 0, 255).astype(np.uint8)


def manual_median_filter(image, kernel_size=3, pad_mode="edge"):
    """Застосовує ручний нелінійний медіанний фільтр із вікном kernel_size x kernel_size."""
    _validate_kernel_size(kernel_size)
    arr = to_grayscale(image).astype(np.float64)
    if kernel_size == 1:
        return arr.astype(np.uint8)

    height, width = arr.shape
    radius = kernel_size // 2
    padded = pad_image(arr, radius, radius, mode=pad_mode)

    neighbors = []
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            neighbors.append(padded[ky : ky + height, kx : kx + width])

    neighborhood_stack = np.stack(neighbors, axis=0)
    median_values = np.median(neighborhood_stack, axis=0)
    return np.clip(np.rint(median_values), 0, 255).astype(np.uint8)


def manual_sobel_filter(image, edge_threshold=None, pad_mode="edge"):
    """Виділяє контури вручну за допомогою горизонтального та вертикального ядер Собеля."""
    grad_x = manual_convolve2d(image, SOBEL_KERNEL_X, pad_mode=pad_mode)
    grad_y = manual_convolve2d(image, SOBEL_KERNEL_Y, pad_mode=pad_mode)
    magnitude = np.hypot(grad_x, grad_y)

    if edge_threshold is not None:
        if not 0 <= edge_threshold <= 255:
            raise ValueError("Поріг контурів edge_threshold повинен бути у [0, 255].")
        return np.where(magnitude >= edge_threshold, 255, 0).astype(np.uint8)

    return np.clip(np.rint(magnitude), 0, 255).astype(np.uint8)


def compute_lbp_map(image, pad_mode="edge"):
    """Обчислює карту локальних бінарних шаблонів (LBP 3x3) для аналізу текстур вручну."""
    gray = to_grayscale(image).astype(np.int32)
    height, width = gray.shape
    padded = pad_image(gray, 1, 1, mode=pad_mode).astype(np.int32)
    center = gray

    # 8 сусідів навколо центрального пікселя за годинниковою стрілкою
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    lbp_map = np.zeros((height, width), dtype=np.uint8)
    for bit_idx, (dy, dx) in enumerate(offsets):
        y_start = 1 + dy
        x_start = 1 + dx
        neighbor = padded[y_start : y_start + height, x_start : x_start + width]
        lbp_map += ((neighbor >= center).astype(np.uint8) << bit_idx)

    return lbp_map
