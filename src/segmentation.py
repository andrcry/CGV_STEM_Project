"""Методи порогової сегментації зображень (ручна реалізація)."""

import numpy as np
from src.utils import to_grayscale


def global_threshold(image, threshold=127, max_val=255, invert=False):
    """Виконує глобальну порогову бінаризацію (на базі формули з CGV_4.py)."""
    if not 0 <= threshold <= 255:
        raise ValueError(f"Поріг бінаризації має бути у [0, 255], отримано: {threshold}")
    if not 1 <= max_val <= 255:
        raise ValueError(f"Максимальне значення max_val має бути у [1, 255], отримано: {max_val}")

    gray = to_grayscale(image)
    if invert:
        return np.where(gray <= threshold, max_val, 0).astype(np.uint8)
    return np.where(gray > threshold, max_val, 0).astype(np.uint8)


def otsu_threshold_value(image):
    """Знаходить оптимальний поріг бінаризації методом Оцу (максимум міжкласової дисперсії)."""
    gray = to_grayscale(image)
    if gray.size == 0:
        raise ValueError("Порожнє зображення неможливо сегментувати.")

    probs = np.bincount(gray.ravel(), minlength=256).astype(np.float64) / float(gray.size)
    total_mean = float(np.sum(np.arange(256, dtype=np.float64) * probs))

    best_threshold = int(gray[0, 0])
    max_between_var = -1.0
    weight_bg = 0.0
    sum_bg = 0.0

    for thresh in range(255):
        weight_bg += float(probs[thresh])
        weight_fg = 1.0 - weight_bg
        if weight_bg == 0.0 or weight_fg == 0.0:
            sum_bg += float(thresh * probs[thresh])
            continue

        sum_bg += float(thresh * probs[thresh])
        mean_diff = (sum_bg / weight_bg) - ((total_mean - sum_bg) / weight_fg)
        between_var = weight_bg * weight_fg * (mean_diff ** 2)

        if between_var > max_between_var:
            max_between_var = between_var
            best_threshold = thresh

    return best_threshold


def otsu_threshold(image, max_val=255, invert=False):
    """Сегментує зображення за автоматично знайденим порогом Оцу."""
    threshold = otsu_threshold_value(image)
    binary = global_threshold(image, threshold=threshold, max_val=max_val, invert=invert)
    return binary, threshold


def threshold_segmentation(image, method="otsu", threshold=127, max_val=255, invert=False):
    """Універсальна обгортка порогової сегментації (повертає маску та використаний поріг)."""
    if method == "global":
        binary = global_threshold(image, threshold=threshold, max_val=max_val, invert=invert)
        return binary, int(threshold)
    if method == "otsu":
        return otsu_threshold(image, max_val=max_val, invert=invert)

    raise ValueError(f"Невідомий метод сегментації: {method}. Очікується 'global' або 'otsu'.")
