"""Допоміжні утиліти для завантаження зображень, шуму, метрик якості та аналізу текстур.

Модуль базується на функціях із попередніх лабораторних робіт (CGV_4.py, CGV_5.py, CGV_6.py)
та розширює їх метриками PSNR, SSIM і текстурними дескрипторами GLCM.
"""

import os
import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


def load_image(img_path, resize_to=None):
    """Завантажує зображення з диска у масив NumPy (uint8) з обробкою крайніх випадків."""
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Файл зображення не знайдено: {img_path}")
    if os.path.getsize(img_path) == 0:
        raise ValueError(f"Файл зображення порожній (0 байтів): {img_path}")

    try:
        with Image.open(img_path) as img:
            img.load()
            if img.mode in ("L", "1"):
                prepared = img.convert("L")
            elif img.mode == "LA":
                prepared = img.convert("L")
            elif img.mode in ("RGBA", "PA"):
                # Композит на біле тло прибирає прозорість без артефактів альфа-каналу
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                composed = Image.alpha_composite(background, img.convert("RGBA"))
                prepared = composed.convert("RGB")
            else:
                prepared = img.convert("RGB")

            if resize_to is not None:
                prepared = prepared.resize(resize_to, Image.Resampling.LANCZOS)

            arr = np.asarray(prepared, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(
            f"Неможливо прочитати пошкоджене або некоректне зображення: {img_path}"
        ) from exc

    if arr.size == 0:
        raise ValueError(f"Зображення не містить пікселів: {img_path}")
    return arr


def save_image(image_array, out_path):
    """Зберігає двовимірний або тривимірний масив зображення у файл."""
    if not isinstance(image_array, np.ndarray) or image_array.size == 0:
        raise ValueError("Неможливо зберегти порожній масив зображення.")

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    clipped = np.clip(image_array, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(out_path)


def to_grayscale(image_array):
    """Перетворює RGB/RGBA зображення у відтінки сірого (формула яскравості з CGV_5.py)."""
    arr = np.asarray(image_array)
    if arr.size == 0:
        raise ValueError("Вхідний масив зображення порожній.")

    if arr.ndim == 2:
        return np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.clip(arr[:, :, 0], 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr_f = arr.astype(np.float64)
        r = arr_f[:, :, 0]
        g = arr_f[:, :, 1]
        b = arr_f[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(np.rint(gray), 0, 255).astype(np.uint8)

    raise ValueError(f"Непідтримувана розмірність масиву зображення: {arr.shape}")


def add_salt_and_pepper_noise(image_array, amount=0.03, seed=42):
    """Додає імпульсний шум 'сіль і перець' із фіксованим seed (на базі CGV_5/CGV_6)."""
    if not 0.0 <= amount <= 1.0:
        raise ValueError("Частка шуму amount повинна бути в діапазоні [0.0, 1.0].")

    gray = to_grayscale(image_array)
    noisy = gray.copy()
    if amount == 0.0:
        return noisy

    rng = np.random.default_rng(seed)
    rand_mat = rng.random(gray.shape)
    noisy[rand_mat < (amount / 2.0)] = 0
    noisy[rand_mat > 1.0 - (amount / 2.0)] = 255
    return noisy


def add_gaussian_noise(image_array, sigma=20.0, seed=42):
    """Додає адитивний гаусів шум із нульовим середнім та заданим sigma (на базі CGV_6)."""
    if sigma < 0.0:
        raise ValueError("Стандартне відхилення sigma не може бути від'ємним.")

    gray = to_grayscale(image_array)
    if sigma == 0.0:
        return gray.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=gray.shape)
    noisy = gray.astype(np.float64) + noise
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _validate_pair(img1, img2):
    """Перевіряє коректність та однаковість розмірів пари зображень для метрик."""
    arr1 = np.asarray(img1, dtype=np.float64)
    arr2 = np.asarray(img2, dtype=np.float64)
    if arr1.size == 0 or arr2.size == 0:
        raise ValueError("Одне з зображень для обчислення метрик порожнє.")
    if arr1.shape != arr2.shape:
        raise ValueError(f"Розміри зображень не збігаються: {arr1.shape} != {arr2.shape}")
    return arr1, arr2


def calc_mse(img1, img2):
    """Обчислює середньоквадратичну похибку MSE між двома зображеннями (з CGV_6.py)."""
    arr1, arr2 = _validate_pair(img1, img2)
    diff = arr1 - arr2
    return float(np.mean(diff ** 2))


def calc_psnr(img1, img2, max_val=255.0):
    """Обчислює пікове відношення сигнал/шум PSNR у децибелах."""
    if max_val <= 0:
        raise ValueError("Параметр max_val повинен бути додатним.")
    mse = calc_mse(img1, img2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def _box_mean_valid(arr, window_size):
    """Обчислює локальні середні через інтегральне зображення без зовнішніх фільтрів."""
    height, width = arr.shape
    sat = np.zeros((height + 1, width + 1), dtype=np.float64)
    sat[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    sums = (
        sat[window_size:, window_size:]
        - sat[:-window_size, window_size:]
        - sat[window_size:, :-window_size]
        + sat[:-window_size, :-window_size]
    )
    return sums / float(window_size * window_size)


def _global_ssim(arr1, arr2, c1, c2):
    """Глобальний розрахунок SSIM для малих матриць (напр. 3x3 або 5x5)."""
    mu_x = float(np.mean(arr1))
    mu_y = float(np.mean(arr2))
    sigma_x2 = float(np.var(arr1))
    sigma_y2 = float(np.var(arr2))
    sigma_xy = float(np.mean((arr1 - mu_x) * (arr2 - mu_y)))
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(numerator / denominator)


def calc_ssim(img1, img2, max_val=255.0, window_size=7):
    """Обчислює індекс структурної подібності SSIM вручну (Wang et al., 2004)."""
    arr1, arr2 = _validate_pair(img1, img2)
    if arr1.ndim != 2:
        arr1 = to_grayscale(arr1).astype(np.float64)
        arr2 = to_grayscale(arr2).astype(np.float64)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    if min(arr1.shape) < window_size:
        return _global_ssim(arr1, arr2, c1, c2)

    mu_x = _box_mean_valid(arr1, window_size)
    mu_y = _box_mean_valid(arr2, window_size)
    sigma_x2 = np.maximum(_box_mean_valid(arr1 * arr1, window_size) - mu_x ** 2, 0.0)
    sigma_y2 = np.maximum(_box_mean_valid(arr2 * arr2, window_size) - mu_y ** 2, 0.0)
    sigma_xy = _box_mean_valid(arr1 * arr2, window_size) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(numerator / denominator))


def calc_image_stats(image_array):
    """Обчислює базові статистичні характеристики: середнє, дисперсію, std та ентропію."""
    gray = to_grayscale(image_array)
    gray_f = gray.astype(np.float64)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probs = hist / float(gray.size)
    nonzero = probs[probs > 0.0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    return {
        "mean": round(float(np.mean(gray_f)), 4),
        "variance": round(float(np.var(gray_f)), 4),
        "std": round(float(np.std(gray_f)), 4),
        "entropy": round(entropy, 4),
    }


def compute_glcm_features(image_array, levels=16):
    """Обчислює текстурні характеристики GLCM (контраст, гомогенність, енергія) вручну."""
    if levels < 2 or levels > 256:
        raise ValueError("Кількість рівнів квантування levels має бути у межах [2, 256].")

    gray = to_grayscale(image_array)
    if gray.shape[1] < 2:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 1.0, "glcm_energy": 1.0}

    # Квантування рівнів яскравості [0..255] -> [0..levels-1]
    quantized = np.minimum((gray.astype(np.int32) * levels) // 256, levels - 1)
    left = quantized[:, :-1].ravel()
    right = quantized[:, 1:].ravel()

    # Симетрична матриця спільної появи для горизонтального зміщення dx=1, dy=0
    glcm = (
        np.bincount(left * levels + right, minlength=levels * levels)
        + np.bincount(right * levels + left, minlength=levels * levels)
    ).reshape((levels, levels)).astype(np.float64)

    total = float(np.sum(glcm))
    if total == 0.0:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 1.0, "glcm_energy": 1.0}

    prob = glcm / total
    idx_i, idx_j = np.indices((levels, levels))
    diff = idx_i - idx_j

    return {
        "glcm_contrast": round(float(np.sum((diff ** 2) * prob)), 4),
        "glcm_homogeneity": round(float(np.sum(prob / (1.0 + np.abs(diff)))), 4),
        "glcm_energy": round(float(np.sum(prob ** 2)), 4),
    }
