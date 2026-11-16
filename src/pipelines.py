"""Реалізація конвеєрів (пайплайнів) обробки зображень та їх порівняння."""

import os
import numpy as np

from src.filters import (
    compute_lbp_map,
    manual_gaussian_filter,
    manual_median_filter,
    manual_sobel_filter,
)
from src.segmentation import threshold_segmentation
from src.utils import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    calc_image_stats,
    calc_mse,
    calc_psnr,
    calc_ssim,
    compute_glcm_features,
    save_image,
    to_grayscale,
)


def _safe_metric(val):
    """Форматує метрики для JSON/CSV (захищаючи від нескінченності у стандартному JSON)."""
    if np.isinf(val):
        return 100.0
    return round(float(val), 4)


def _apply_noise(gray_img, noise_cfg):
    """Додає обраний тип шуму з фіксованим seed згідно з конфігурацією."""
    noise_type = noise_cfg.get("type", "salt_and_pepper")
    seed = int(noise_cfg.get("seed", 42))

    if noise_type == "none":
        return gray_img.copy()
    if noise_type == "salt_and_pepper":
        amount = float(noise_cfg.get("amount", 0.05))
        return add_salt_and_pepper_noise(gray_img, amount=amount, seed=seed)
    if noise_type == "gaussian":
        sigma = float(noise_cfg.get("sigma", 20.0))
        return add_gaussian_noise(gray_img, sigma=sigma, seed=seed)

    raise ValueError(f"Невідомий тип шуму: {noise_type}")


def _apply_configured_filter(image, filter_cfg):
    """Викликає ручний фільтр за параметрами конфігурації."""
    filter_type = filter_cfg.get("filter_type", "median")
    kernel_size = int(filter_cfg.get("kernel_size", 3))
    pad_mode = filter_cfg.get("pad_mode", "edge")

    if filter_type == "gaussian":
        sigma = float(filter_cfg.get("sigma", 1.0))
        return manual_gaussian_filter(
            image, kernel_size=kernel_size, sigma=sigma, pad_mode=pad_mode
        )
    if filter_type == "median":
        return manual_median_filter(image, kernel_size=kernel_size, pad_mode=pad_mode)

    raise ValueError(f"Невідомий тип фільтра: {filter_type}")


def _pair_metrics(ref_img, target_img):
    """Обчислює базовий набір метрик якості (MSE, PSNR, SSIM) для пари зображень."""
    return {
        "mse": _safe_metric(calc_mse(ref_img, target_img)),
        "psnr": _safe_metric(calc_psnr(ref_img, target_img)),
        "ssim": _safe_metric(calc_ssim(ref_img, target_img)),
    }


def _save_stage_outputs(output_dir, pipeline_folder, image_name, stage_map):
    """Зберігає проміжні та кінцеві результати стадій у папку пайплайна."""
    img_out_dir = os.path.join(output_dir, pipeline_folder, image_name)
    for filename, img_arr in stage_map.items():
        save_image(img_arr, os.path.join(img_out_dir, f"{filename}.png"))


def run_pipeline_1(image, config, image_name="image", output_dir=None):
    """Пайплайн 1: Grayscale -> (Шум) -> Ручний фільтр -> Текстурний аналіз (GLCM + LBP)."""
    p1_cfg = config.get("pipeline_1", {})
    noise_cfg = p1_cfg.get("noise", {"type": "salt_and_pepper", "amount": 0.05, "seed": 42})
    filter_cfg = p1_cfg.get("filter", {"filter_type": "median", "kernel_size": 3})

    # Стадія 1 (Колірне перетворення): RGB -> градації сірого
    stage1_gray = to_grayscale(image)
    # Тестове спотворення входу (моделювання зашумлення для оцінки шумозаглушення)
    stage2_noisy = _apply_noise(stage1_gray, noise_cfg)
    # Стадія 2 (Фільтрація зображень): ручний медіанний або гаусів фільтр
    stage3_filtered = _apply_configured_filter(stage2_noisy, filter_cfg)
    # Стадія 3 (Аналіз текстур): локальні бінарні шаблони LBP та матриця GLCM
    stage4_lbp = compute_lbp_map(stage3_filtered, pad_mode=filter_cfg.get("pad_mode", "edge"))

    stages = {
        "stage1_gray": stage1_gray,
        "stage2_noisy": stage2_noisy,
        "stage3_filtered": stage3_filtered,
        "stage4_texture_lbp": stage4_lbp,
    }
    if output_dir and bool(config.get("save_intermediate", True)):
        _save_stage_outputs(output_dir, "pipeline_1", image_name, stages)

    noisy_metrics = _pair_metrics(stage1_gray, stage2_noisy)
    filtered_metrics = _pair_metrics(stage1_gray, stage3_filtered)
    filtered_metrics["psnr_gain"] = round(
        float(filtered_metrics["psnr"] - noisy_metrics["psnr"]), 4
    )

    metrics = {
        "noisy_vs_gray": noisy_metrics,
        "filtered_vs_gray": filtered_metrics,
        "texture_gray": {**calc_image_stats(stage1_gray), **compute_glcm_features(stage1_gray)},
        "texture_filtered": {
            **calc_image_stats(stage3_filtered),
            **compute_glcm_features(stage3_filtered),
        },
    }
    return {"stages": stages, "metrics": metrics}


def run_pipeline_2(image, config, image_name="image", output_dir=None):
    """Пайплайн 2: Grayscale -> Порогова сегментація -> Оператор Собеля (контурний аналіз)."""
    p2_cfg = config.get("pipeline_2", {})
    seg_cfg = p2_cfg.get("segmentation", {"method": "otsu", "threshold": 127})
    sobel_cfg = p2_cfg.get("sobel", {"edge_threshold": None, "pad_mode": "edge"})

    # Стадія 1 (Колірне перетворення): RGB -> градації сірого
    stage1_gray = to_grayscale(image)
    # Стадія 2 (Сегментація зображень): порогова бінаризація (глобальна або Оцу)
    stage2_segmented, used_threshold = threshold_segmentation(
        stage1_gray,
        method=seg_cfg.get("method", "otsu"),
        threshold=int(seg_cfg.get("threshold", 127)),
        invert=bool(seg_cfg.get("invert", False)),
    )
    # Стадія 3 (Контурний аналіз): ручний оператор Собеля
    stage3_sobel = manual_sobel_filter(
        stage2_segmented,
        edge_threshold=sobel_cfg.get("edge_threshold"),
        pad_mode=sobel_cfg.get("pad_mode", "edge"),
    )
    sobel_raw_gray = manual_sobel_filter(
        stage1_gray,
        edge_threshold=sobel_cfg.get("edge_threshold"),
        pad_mode=sobel_cfg.get("pad_mode", "edge"),
    )

    stages_to_save = {
        "stage1_gray": stage1_gray,
        "stage2_segmented": stage2_segmented,
        "stage3_sobel": stage3_sobel,
    }
    if output_dir and bool(config.get("save_intermediate", True)):
        _save_stage_outputs(output_dir, "pipeline_2", image_name, stages_to_save)

    metrics = {
        "used_threshold": int(used_threshold),
        "foreground_ratio_pct": round(float(np.mean(stage2_segmented > 0) * 100.0), 2),
        "edge_density_segmented_pct": round(float(np.mean(stage3_sobel > 30) * 100.0), 2),
        "edge_density_raw_gray_pct": round(float(np.mean(sobel_raw_gray > 30) * 100.0), 2),
        "segmented_vs_gray": _pair_metrics(stage1_gray, stage2_segmented),
        "sobel_vs_raw_sobel": _pair_metrics(sobel_raw_gray, stage3_sobel),
    }
    return {"stages": {**stages_to_save, "sobel_raw_gray": sobel_raw_gray}, "metrics": metrics}


def run_ablation_for_image(image, seed=42):
    """Порівнює методи фільтрації та сегментації на одному зображенні при різних умовах."""
    gray = to_grayscale(image)
    ablation = {}

    noise_scenarios = {
        "sp_0.05": add_salt_and_pepper_noise(gray, amount=0.05, seed=seed),
        "sp_0.15": add_salt_and_pepper_noise(gray, amount=0.15, seed=seed),
        "gauss_15": add_gaussian_noise(gray, sigma=15.0, seed=seed),
        "gauss_30": add_gaussian_noise(gray, sigma=30.0, seed=seed),
    }

    for noise_name, noisy_img in noise_scenarios.items():
        variants = {
            "noisy": noisy_img,
            "gaussian_3x3": manual_gaussian_filter(noisy_img, kernel_size=3, sigma=1.0),
            "gaussian_5x5": manual_gaussian_filter(noisy_img, kernel_size=5, sigma=1.4),
            "median_3x3": manual_median_filter(noisy_img, kernel_size=3),
            "median_5x5": manual_median_filter(noisy_img, kernel_size=5),
        }
        ablation[noise_name] = {
            v_name: {
                **_pair_metrics(gray, v_img),
                "glcm_contrast": compute_glcm_features(v_img)["glcm_contrast"],
            }
            for v_name, v_img in variants.items()
        }

    # Порівняння глобального порогу 127 та адаптивного порогу Оцу
    seg_global, _ = threshold_segmentation(gray, method="global", threshold=127)
    seg_otsu, otsu_t = threshold_segmentation(gray, method="otsu")
    ablation["segmentation"] = {
        "global_127": {
            "threshold": 127,
            "ssim_vs_gray": _safe_metric(calc_ssim(gray, seg_global)),
            "edge_density_pct": round(
                float(np.mean(manual_sobel_filter(seg_global) > 30) * 100.0), 2
            ),
        },
        "otsu": {
            "threshold": int(otsu_t),
            "ssim_vs_gray": _safe_metric(calc_ssim(gray, seg_otsu)),
            "edge_density_pct": round(
                float(np.mean(manual_sobel_filter(seg_otsu) > 30) * 100.0), 2
            ),
        },
    }
    return ablation
