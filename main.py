"""Головний скрипт запуску STEM-проєкту через конфігурацію та параметри CLI."""

import argparse
import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from src.pipelines import run_ablation_for_image, run_pipeline_1, run_pipeline_2
from src.utils import SUPPORTED_EXTENSIONS, load_image


def _infer_image_group(filename):
    """Визначає семантичну групу зображення за числовим префіксом у вибірці."""
    stem = os.path.splitext(filename)[0]
    prefix = stem.split("_", 1)[0]
    if prefix.isdigit():
        idx = int(prefix)
        if 1 <= idx <= 10:
            return "natural_texture"
        if 11 <= idx <= 20:
            return "urban_geometric"
        if 21 <= idx <= 30:
            return "text_contrast"
    return "custom"


def _load_config(config_path):
    """Зчитує параметри алгоритмів і пайплайнів із JSON-конфігурації."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Конфігураційний файл не знайдено: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_cli_overrides(config, args):
    """Перевизначає параметри конфігурації значеннями з аргументів командного рядка."""
    top_overrides = {
        "input_dir": args.input,
        "output_dir": args.output,
        "run_ablation": False if args.no_ablation else None,
        "generate_charts": False if args.no_charts else None,
    }
    for key, val in top_overrides.items():
        if val is not None:
            config[key] = val

    p1_cfg = config.setdefault("pipeline_1", {})
    noise_cfg = p1_cfg.setdefault("noise", {})
    filter_cfg = p1_cfg.setdefault("filter", {})
    for key, val in (
        ("type", args.noise_type),
        ("amount", args.noise_amount),
    ):
        if val is not None:
            noise_cfg[key] = val

    for key, val in (
        ("filter_type", args.filter),
        ("kernel_size", args.kernel_size),
        ("sigma", args.sigma),
    ):
        if val is not None:
            filter_cfg[key] = val

    seg_cfg = config.setdefault("pipeline_2", {}).setdefault("segmentation", {})
    if args.threshold_method is not None:
        seg_cfg["method"] = args.threshold_method
    if args.threshold is not None:
        seg_cfg["threshold"] = args.threshold


def _avg_group(subset):
    """Обчислює середні метрики для заданої підмножини записів."""
    p1_items = [r["pipeline_1"] for r in subset if "pipeline_1" in r]
    p2_items = [r["pipeline_2"] for r in subset if "pipeline_2" in r]
    summary = {"count": len(subset)}

    if p1_items:
        summary["pipeline_1_avg"] = {
            "noisy_psnr": round(float(np.mean([x["noisy_vs_gray"]["psnr"] for x in p1_items])), 4),
            "filtered_psnr": round(
                float(np.mean([x["filtered_vs_gray"]["psnr"] for x in p1_items])), 4
            ),
            "filtered_ssim": round(
                float(np.mean([x["filtered_vs_gray"]["ssim"] for x in p1_items])), 4
            ),
            "psnr_gain": round(
                float(np.mean([x["filtered_vs_gray"]["psnr_gain"] for x in p1_items])), 4
            ),
            "glcm_contrast_gray": round(
                float(np.mean([x["texture_gray"]["glcm_contrast"] for x in p1_items])), 4
            ),
            "glcm_contrast_filtered": round(
                float(np.mean([x["texture_filtered"]["glcm_contrast"] for x in p1_items])), 4
            ),
        }

    if p2_items:
        summary["pipeline_2_avg"] = {
            "used_threshold": round(float(np.mean([x["used_threshold"] for x in p2_items])), 2),
            "foreground_ratio_pct": round(
                float(np.mean([x["foreground_ratio_pct"] for x in p2_items])), 2
            ),
            "edge_density_segmented_pct": round(
                float(np.mean([x["edge_density_segmented_pct"] for x in p2_items])), 2
            ),
            "edge_density_raw_gray_pct": round(
                float(np.mean([x["edge_density_raw_gray_pct"] for x in p2_items])), 2
            ),
            "segmented_ssim": round(
                float(np.mean([x["segmented_vs_gray"]["ssim"] for x in p2_items])), 4
            ),
        }
    return summary


def _summarize_records(records):
    """Обчислює середні метрики за всією вибіркою та окремими групами зображень."""
    if not records:
        return {}
    by_group = {
        grp: _avg_group([r for r in records if r["group"] == grp])
        for grp in sorted({r["group"] for r in records})
    }
    return {"overall": _avg_group(records), "by_group": by_group}


def _summarize_ablation(ablation_rows):
    """Агрегує результати порівняльних експериментів (Gaussian vs Median, Global vs Otsu)."""
    if not ablation_rows:
        return {}

    agg = {}
    for n_key in ("sp_0.05", "sp_0.15", "gauss_15", "gauss_30"):
        agg[n_key] = {}
        for f_key in ("noisy", "gaussian_3x3", "gaussian_5x5", "median_3x3", "median_5x5"):
            items = [row[n_key][f_key] for row in ablation_rows if n_key in row]
            agg[n_key][f_key] = {
                "mse": round(float(np.mean([x["mse"] for x in items])), 2),
                "psnr": round(float(np.mean([x["psnr"] for x in items])), 2),
                "ssim": round(float(np.mean([x["ssim"] for x in items])), 4),
                "glcm_contrast": round(float(np.mean([x["glcm_contrast"] for x in items])), 4),
            }

    seg_global = [row["segmentation"]["global_127"] for row in ablation_rows]
    seg_otsu = [row["segmentation"]["otsu"] for row in ablation_rows]
    agg["segmentation"] = {
        "global_127": {
            "ssim_vs_gray": round(float(np.mean([x["ssim_vs_gray"] for x in seg_global])), 4),
            "edge_density_pct": round(
                float(np.mean([x["edge_density_pct"] for x in seg_global])), 2
            ),
        },
        "otsu": {
            "avg_threshold": round(float(np.mean([x["threshold"] for x in seg_otsu])), 2),
            "ssim_vs_gray": round(float(np.mean([x["ssim_vs_gray"] for x in seg_otsu])), 4),
            "edge_density_pct": round(
                float(np.mean([x["edge_density_pct"] for x in seg_otsu])), 2
            ),
        },
    }
    return agg


def _export_csv(records, csv_path):
    """Зберігає зведену таблицю метрик кожного зображення у формат CSV."""
    fieldnames = [
        "image", "group", "p1_noisy_mse", "p1_noisy_psnr", "p1_noisy_ssim",
        "p1_filtered_mse", "p1_filtered_psnr", "p1_filtered_ssim", "p1_psnr_gain",
        "p1_glcm_contrast_gray", "p1_glcm_contrast_filtered", "p2_used_threshold",
        "p2_fg_ratio_pct", "p2_seg_ssim", "p2_edge_density_seg_pct", "p2_edge_density_raw_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            p1 = row.get("pipeline_1", {})
            p2 = row.get("pipeline_2", {})
            writer.writerow(
                {
                    "image": row["image"],
                    "group": row["group"],
                    "p1_noisy_mse": p1.get("noisy_vs_gray", {}).get("mse", ""),
                    "p1_noisy_psnr": p1.get("noisy_vs_gray", {}).get("psnr", ""),
                    "p1_noisy_ssim": p1.get("noisy_vs_gray", {}).get("ssim", ""),
                    "p1_filtered_mse": p1.get("filtered_vs_gray", {}).get("mse", ""),
                    "p1_filtered_psnr": p1.get("filtered_vs_gray", {}).get("psnr", ""),
                    "p1_filtered_ssim": p1.get("filtered_vs_gray", {}).get("ssim", ""),
                    "p1_psnr_gain": p1.get("filtered_vs_gray", {}).get("psnr_gain", ""),
                    "p1_glcm_contrast_gray": p1.get("texture_gray", {}).get("glcm_contrast", ""),
                    "p1_glcm_contrast_filtered": p1.get("texture_filtered", {}).get(
                        "glcm_contrast", ""
                    ),
                    "p2_used_threshold": p2.get("used_threshold", ""),
                    "p2_fg_ratio_pct": p2.get("foreground_ratio_pct", ""),
                    "p2_seg_ssim": p2.get("segmented_vs_gray", {}).get("ssim", ""),
                    "p2_edge_density_seg_pct": p2.get("edge_density_segmented_pct", ""),
                    "p2_edge_density_raw_pct": p2.get("edge_density_raw_gray_pct", ""),
                }
            )


def _plot_ablation_bars(ablation_summary, charts_dir):
    """Будує порівняльну діаграму PSNR та SSIM для різних фільтрів і шумів."""
    scenarios = ["sp_0.05", "sp_0.15", "gauss_15", "gauss_30"]
    labels = ["Сіль-перець 5%", "Сіль-перець 15%", "Гаусів σ=15", "Гаусів σ=30"]
    methods = (
        ("noisy", "Без фільтра"),
        ("gaussian_3x3", "Гаус 3x3"),
        ("gaussian_5x5", "Гаус 5x5"),
        ("median_3x3", "Медіанний 3x3"),
        ("median_5x5", "Медіанний 5x5"),
    )

    x_pos = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, (m_key, m_label) in enumerate(methods):
        offset = (idx - 2) * 0.15
        axes[0].bar(
            x_pos + offset,
            [ablation_summary[s][m_key]["psnr"] for s in scenarios],
            0.15,
            label=m_label,
        )
        axes[1].bar(
            x_pos + offset,
            [ablation_summary[s][m_key]["ssim"] for s in scenarios],
            0.15,
            label=m_label,
        )

    for ax, title, ylabel in (
        (axes[0], "Середній PSNR (дБ) за типами шуму", "PSNR (дБ)"),
        (axes[1], "Середній SSIM за типами шуму", "SSIM"),
    ):
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "ablation_psnr_ssim.png"), dpi=150)
    plt.close(fig)


def _plot_edge_density_bars(summary_data, charts_dir):
    """Будує діаграму впливу попередньої сегментації на щільність контурів Собеля."""
    by_group = summary_data.get("by_group", {})
    groups = [g for g in by_group if "pipeline_2_avg" in by_group[g]]
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_g = np.arange(len(groups))
    seg_dens = [by_group[g]["pipeline_2_avg"]["edge_density_segmented_pct"] for g in groups]
    raw_dens = [by_group[g]["pipeline_2_avg"]["edge_density_raw_gray_pct"] for g in groups]
    ax.bar(x_g - 0.18, raw_dens, 0.35, label="Собель на сирому grayscale")
    ax.bar(x_g + 0.18, seg_dens, 0.35, label="Пайплайн 2 (Оцу -> Собель)")
    ax.set_xticks(x_g)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Частка контурних пікселів (%)")
    ax.set_title("Вплив попередньої сегментації на щільність контурів Собеля")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "pipeline2_edge_density.png"), dpi=150)
    plt.close(fig)


def _render_montage_grid(samples, pipe_key, columns, out_path):
    """Візуалізує сітку проміжних стадій для репрезентативних зображень."""
    selected = samples[:3]
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(12, 3 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, item in enumerate(selected):
        stages = item[pipe_key]["stages"]
        for col_idx, (stage_key, stage_title) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.imshow(stages[stage_key], cmap="gray", vmin=0, vmax=255)
            if row_idx == 0:
                ax.set_title(stage_title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(item["name"], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _generate_all_charts(ablation_summary, summary, showcase_samples, output_dir):
    """Генерує всі діаграми метрик та візуальні колажі для звіту."""
    plt.switch_backend("Agg")
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    if ablation_summary:
        _plot_ablation_bars(ablation_summary, charts_dir)
    _plot_edge_density_bars(summary, charts_dir)

    if showcase_samples and all("p1" in s for s in showcase_samples):
        p1_cols = [
            ("stage1_gray", "1. Grayscale"),
            ("stage2_noisy", "2. Зашумлене"),
            ("stage3_filtered", "3. Відфільтроване"),
            ("stage4_texture_lbp", "4. Текстура (LBP)"),
        ]
        _render_montage_grid(
            showcase_samples, "p1", p1_cols, os.path.join(charts_dir, "visual_stages_pipeline1.png")
        )

    if showcase_samples and all("p2" in s for s in showcase_samples):
        p2_cols = [
            ("stage1_gray", "1. Grayscale"),
            ("stage2_segmented", "2. Сегментація"),
            ("stage3_sobel", "3. Собель (Пайплайн 2)"),
            ("sobel_raw_gray", "Порівняння: Собель (raw)"),
        ]
        _render_montage_grid(
            showcase_samples, "p2", p2_cols, os.path.join(charts_dir, "visual_stages_pipeline2.png")
        )


def _process_single_image(filename, input_dir, output_dir, config, pipeline_mode):
    """Завантажує одне зображення та запускає вибрані пайплайни і порівняння."""
    img_stem = os.path.splitext(filename)[0]
    image = load_image(os.path.join(input_dir, filename))
    entry = {"image": filename, "group": _infer_image_group(filename)}
    bundle = {"name": img_stem}

    if pipeline_mode in ("1", "all"):
        res1 = run_pipeline_1(image, config, image_name=img_stem, output_dir=output_dir)
        entry["pipeline_1"] = res1["metrics"]
        bundle["p1"] = res1
    if pipeline_mode in ("2", "all"):
        res2 = run_pipeline_2(image, config, image_name=img_stem, output_dir=output_dir)
        entry["pipeline_2"] = res2["metrics"]
        bundle["p2"] = res2

    ablation_row = None
    if config.get("run_ablation", True):
        seed = int(config.get("pipeline_1", {}).get("noise", {}).get("seed", 42))
        ablation_row = run_ablation_for_image(image, seed=seed)

    return entry, bundle, ablation_row


SHOWCASE_NAMES = {"01_baboon", "11_building", "21_sudoku"}


def _process_dataset(files, input_dir, output_dir, config, pipeline_mode):
    """Проганяє вибрані пайплайни та порівняльні тести по всій вибірці зображень."""
    records, ablation_rows, skipped, showcase = [], [], [], []
    has_showcase = any(os.path.splitext(f)[0] in SHOWCASE_NAMES for f in files)

    for filename in files:
        try:
            entry, bundle, ablation_row = _process_single_image(
                filename, input_dir, output_dir, config, pipeline_mode
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[!] Пропущено файл {filename}: {exc}")
            skipped.append({"image": filename, "error": str(exc)})
            continue

        records.append(entry)
        if bundle["name"] in SHOWCASE_NAMES or (not has_showcase and len(showcase) < 3):
            showcase.append(bundle)
        if ablation_row is not None:
            ablation_rows.append(ablation_row)

    return records, ablation_rows, skipped, showcase[:3]


def _print_console_summary(records, skipped, summary, json_path, csv_path):
    """Виводить стислі підсумки роботи пайплайнів у консоль."""
    print(f"[+] Успішно оброблено зображень: {len(records)} (пропущено: {len(skipped)})")
    print(f"[+] Метрики збережено у {json_path} та {csv_path}")
    overall = summary.get("overall", {})
    if "pipeline_1_avg" in overall:
        p1_avg = overall["pipeline_1_avg"]
        print(
            f"[Пайплайн 1] PSNR: {p1_avg['noisy_psnr']} -> {p1_avg['filtered_psnr']} дБ "
            f"(+{p1_avg['psnr_gain']} дБ), SSIM: {p1_avg['filtered_ssim']}"
        )
    if "pipeline_2_avg" in overall:
        p2_avg = overall["pipeline_2_avg"]
        print(
            f"[Пайплайн 2] Сер. поріг: {p2_avg['used_threshold']}, "
            f"щільність контурів: {p2_avg['edge_density_segmented_pct']}% "
            f"(raw: {p2_avg['edge_density_raw_gray_pct']}%)"
        )


def parse_args():
    """Розбирає аргументи командного рядка для налаштування запуску."""
    parser = argparse.ArgumentParser(
        description="STEM-проєкт: класичні методи фільтрації, сегментації та аналізу текстур"
    )
    parser.add_argument("--config", default="configs/config.json", help="Шлях до JSON-конфіга")
    parser.add_argument("--pipeline", choices=["1", "2", "all"], default="all", help="Пайплайн")
    parser.add_argument("--input", help="Папка з вхідними зображеннями")
    parser.add_argument("--output", help="Папка для збереження результатів")
    parser.add_argument("--filter", choices=["gaussian", "median"], help="Фільтр Пайплайна 1")
    parser.add_argument("--kernel-size", type=int, help="Розмір вікна фільтра (3, 5, ...)")
    parser.add_argument("--sigma", type=float, help="Sigma для Гаусового фільтра")
    parser.add_argument("--noise-type", choices=["salt_and_pepper", "gaussian", "none"])
    parser.add_argument("--noise-amount", type=float, help="Інтенсивність шуму salt_and_pepper")
    parser.add_argument("--threshold-method", choices=["otsu", "global"], help="Метод сегментації")
    parser.add_argument("--threshold", type=int, help="Поріг для глобальної бінаризації")
    parser.add_argument("--no-ablation", action="store_true", help="Вимкнути порівняльний прогін")
    parser.add_argument("--no-charts", action="store_true", help="Не будувати графіки")
    return parser.parse_args()


def main():
    """Точка входу: завантажує вибірку, виконує пайплайни та зберігає метрики/графіки."""
    args = parse_args()
    config = _load_config(args.config)
    _apply_cli_overrides(config, args)

    input_dir = config.get("input_dir", "data")
    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Папку з вхідними даними не знайдено: {input_dir}")

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS))
    if not files:
        raise RuntimeError(f"У папці {input_dir} не знайдено підтримуваних зображень.")

    records, ablation_rows, skipped, showcase = _process_dataset(
        files, input_dir, output_dir, config, args.pipeline
    )
    summary = _summarize_records(records)
    ablation_summary = _summarize_ablation(ablation_rows)

    json_path = os.path.join(output_dir, "metrics_log.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config,
                "processed_count": len(records),
                "skipped_files": skipped,
                "summary": summary,
                "ablation_summary": ablation_summary,
                "per_image": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    _export_csv(records, csv_path)
    if config.get("generate_charts", True):
        _generate_all_charts(ablation_summary, summary, showcase, output_dir)

    _print_console_summary(records, skipped, summary, json_path, csv_path)


if __name__ == "__main__":
    main()
