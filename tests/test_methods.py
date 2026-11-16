"""Модульні автотести для ручних методів обробки зображень, метрик та пайплайнів.

Перевіряє коректність результатів на малих тестових матрицях 3x3 / 4x4 / 5x5
(критерій C3) та стабільність на крайніх випадках (критерій C2).
"""

import os
import tempfile
import unittest
import numpy as np

from src.filters import (
    build_gaussian_kernel,
    compute_lbp_map,
    manual_convolve2d,
    manual_gaussian_filter,
    manual_median_filter,
    manual_sobel_filter,
)
from src.pipelines import run_pipeline_1, run_pipeline_2
from src.segmentation import global_threshold, otsu_threshold, otsu_threshold_value
from src.utils import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    calc_mse,
    calc_psnr,
    calc_ssim,
    compute_glcm_features,
    load_image,
    to_grayscale,
)


class TestImageProcessingMethods(unittest.TestCase):
    """Тести коректності класичних алгоритмів на контрольних матрицях."""

    def test_to_grayscale_3x3_and_monochrome(self):
        """Перевіряє переведення RGB 3x3 у сірий та збереження 2D монохромного входу."""
        rgb = np.full((3, 3, 3), [100, 150, 200], dtype=np.uint8)
        gray = to_grayscale(rgb)
        # 0.299*100 + 0.587*150 + 0.114*200 = 140.75 -> 141
        self.assertEqual(gray.shape, (3, 3))
        self.assertTrue(np.all(gray == 141))

        mono = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        self.assertTrue(np.array_equal(to_grayscale(mono), mono))

    def test_metrics_3x3_hand_calculated(self):
        """Перевіряє MSE, PSNR, SSIM на точних еталонних матрицях 3x3."""
        img_a = np.full((3, 3), 100, dtype=np.uint8)
        img_b = np.full((3, 3), 110, dtype=np.uint8)

        self.assertAlmostEqual(calc_mse(img_a, img_b), 100.0, places=6)
        expected_psnr = 10.0 * np.log10((255.0 ** 2) / 100.0)
        self.assertAlmostEqual(calc_psnr(img_a, img_b), expected_psnr, places=6)

        self.assertEqual(calc_mse(img_a, img_a), 0.0)
        self.assertTrue(np.isinf(calc_psnr(img_a, img_a)))
        self.assertAlmostEqual(calc_ssim(img_a, img_a), 1.0, places=6)

    def test_manual_convolution_and_gaussian_3x3_5x5(self):
        """Перевіряє ручну 2D-згортку та Гаусовий фільтр на матрицях 3x3 і 5x5."""
        mat_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        scale_kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.float64)
        conv = manual_convolve2d(mat_3x3, scale_kernel)
        self.assertTrue(np.array_equal(conv, mat_3x3 * 2.0))

        # На 5x5 з імпульсом 100 у центрі значення центру дорівнює 100 * kernel[1, 1]
        impulse = np.zeros((5, 5), dtype=np.uint8)
        impulse[2, 2] = 100
        kernel = build_gaussian_kernel(kernel_size=3, sigma=1.0)
        self.assertAlmostEqual(float(np.sum(kernel)), 1.0, places=6)
        blurred = manual_gaussian_filter(impulse, kernel_size=3, sigma=1.0)
        self.assertEqual(int(blurred[2, 2]), int(round(100.0 * kernel[1, 1])))

    def test_manual_median_filter_3x3_and_5x5(self):
        """Перевіряє ручний медіанний фільтр на імпульсному викиді 3x3 та 5x5."""
        mat_3x3 = np.array([[1, 2, 3], [4, 250, 6], [7, 8, 9]], dtype=np.uint8)
        med_3 = manual_median_filter(mat_3x3, kernel_size=3)
        # Відсортований масив [1,2,3,4,6,7,8,9,250] -> медіана у центрі = 6
        self.assertEqual(int(med_3[1, 1]), 6)

        mat_5x5 = np.full((5, 5), 20, dtype=np.uint8)
        mat_5x5[2, 2] = 255
        med_5 = manual_median_filter(mat_5x5, kernel_size=3)
        self.assertTrue(np.all(med_5 == 20))

    def test_manual_sobel_filter_5x5_vertical_edge(self):
        """Перевіряє точне значення градієнта Собеля на вертикальному перепаді 5x5."""
        step_img = np.zeros((5, 5), dtype=np.uint8)
        step_img[:, 2:] = 20
        sobel = manual_sobel_filter(step_img, pad_mode="edge")

        # Для перепаду 0 -> 20 сума ядра Gx праворуч (1 + 2 + 1) * 20 = 80, Gy = 0
        self.assertEqual(int(sobel[2, 1]), 80)
        self.assertEqual(int(sobel[2, 2]), 80)
        self.assertEqual(int(sobel[2, 0]), 0)
        self.assertEqual(int(sobel[2, 4]), 0)

    def test_segmentation_global_and_otsu_4x4(self):
        """Перевіряє глобальну порогову бінаризацію та автоматичний поріг Оцу."""
        mat_4x4 = np.array(
            [
                [10, 20, 200, 210],
                [15, 25, 205, 215],
                [10, 30, 210, 220],
                [20, 30, 220, 230],
            ],
            dtype=np.uint8,
        )
        expected = np.array(
            [
                [0, 0, 255, 255],
                [0, 0, 255, 255],
                [0, 0, 255, 255],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        )
        self.assertTrue(np.array_equal(global_threshold(mat_4x4, threshold=127), expected))

        otsu_bin, otsu_t = otsu_threshold(mat_4x4)
        self.assertTrue(30 <= otsu_t < 200)
        self.assertTrue(np.array_equal(otsu_bin, expected))

    def test_texture_lbp_and_glcm_3x3(self):
        """Перевіряє ручне обчислення LBP-коду та ознак GLCM на матриці 3x3."""
        patch = np.array(
            [
                [10, 50, 50],
                [10, 30, 50],
                [10, 10, 10],
            ],
            dtype=np.uint8,
        )
        lbp = compute_lbp_map(patch)
        # Біти за годинниковою стрілкою з (-1,-1): 0, 1, 1, 1, 0, 0, 0, 0 => 2+4+8 = 14
        self.assertEqual(int(lbp[1, 1]), 14)

        const_img = np.full((3, 3), 120, dtype=np.uint8)
        glcm = compute_glcm_features(const_img)
        self.assertEqual(glcm["glcm_contrast"], 0.0)
        self.assertEqual(glcm["glcm_homogeneity"], 1.0)
        self.assertEqual(glcm["glcm_energy"], 1.0)

    def test_pipelines_end_to_end_reproducible(self):
        """Перевіряє повний цикл Пайплайна 1 та Пайплайна 2 з фіксованим seed."""
        synth = np.zeros((16, 16, 3), dtype=np.uint8)
        synth[4:12, 4:12] = [200, 200, 200]
        cfg = {
            "save_intermediate": False,
            "pipeline_1": {
                "noise": {"type": "salt_and_pepper", "amount": 0.1, "seed": 42},
                "filter": {"filter_type": "median", "kernel_size": 3, "pad_mode": "edge"},
            },
            "pipeline_2": {
                "segmentation": {"method": "otsu", "threshold": 127},
                "sobel": {"edge_threshold": None, "pad_mode": "edge"},
            },
        }
        res1_a = run_pipeline_1(synth, cfg)
        res1_b = run_pipeline_1(synth, cfg)
        self.assertTrue(
            np.array_equal(res1_a["stages"]["stage3_filtered"], res1_b["stages"]["stage3_filtered"])
        )
        self.assertGreater(
            res1_a["metrics"]["filtered_vs_gray"]["psnr"],
            res1_a["metrics"]["noisy_vs_gray"]["psnr"],
        )

        res2 = run_pipeline_2(synth, cfg)
        self.assertEqual(res2["stages"]["stage3_sobel"].shape, (16, 16))
        self.assertGreater(res2["metrics"]["edge_density_segmented_pct"], 0.0)


class TestEdgeCases(unittest.TestCase):
    """Тести обробки крайніх випадків (критерій C2)."""

    def test_tiny_1x1_and_constant_images(self):
        """Перевіряє відсутність збоїв на зображеннях 1x1 та однотонних матрицях."""
        tiny = np.array([[42]], dtype=np.uint8)
        self.assertEqual(manual_gaussian_filter(tiny, kernel_size=3).shape, (1, 1))
        self.assertEqual(manual_median_filter(tiny, kernel_size=3).shape, (1, 1))
        self.assertEqual(int(manual_sobel_filter(tiny)[0, 0]), 0)
        self.assertEqual(compute_lbp_map(tiny).shape, (1, 1))
        self.assertEqual(otsu_threshold_value(tiny), 42)
        self.assertEqual(add_salt_and_pepper_noise(tiny, amount=0.1, seed=1).shape, (1, 1))
        self.assertEqual(add_gaussian_noise(tiny, sigma=10.0, seed=1).shape, (1, 1))
        self.assertAlmostEqual(calc_ssim(tiny, tiny), 1.0, places=6)
        self.assertEqual(compute_glcm_features(tiny)["glcm_contrast"], 0.0)

    def test_invalid_files_and_parameters(self):
        """Перевіряє коректну генерацію винятків на пошкоджених файлах та невірних параметрах."""
        with self.assertRaises(FileNotFoundError):
            load_image("non_existent_file_12345.png")

        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_file = os.path.join(tmp_dir, "empty.png")
            with open(empty_file, "wb"):
                pass
            with self.assertRaises(ValueError):
                load_image(empty_file)

            corrupt_file = os.path.join(tmp_dir, "corrupt.png")
            with open(corrupt_file, "wb") as handle:
                handle.write(b"not a real image")
            with self.assertRaises(ValueError):
                load_image(corrupt_file)

        with self.assertRaises(ValueError):
            manual_median_filter(np.zeros((3, 3), dtype=np.uint8), kernel_size=4)
        with self.assertRaises(ValueError):
            calc_mse(np.zeros((3, 3)), np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
