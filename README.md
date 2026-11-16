# STEM-проєкт: Аналіз та обробка зображень класичними методами комп'ютерної графіки

Програмне рішення для лабораторно-дослідницького STEM-проєкту з дисципліни **«Комп'ютерна графіка»**.  
Тема: *«Аналіз та обробка зображень за допомогою класичних методів фільтрації, сегментації та аналізу текстур»*.

## Структура проєкту

```text
CGV_STEM_Project/
├── configs/
│   └── config.json          # Параметри пайплайнів, фільтрів, порогів і шуму
├── data/                    # Вхідна вибірка зображень (.png / .bmp, 30 шт.)
├── outputs/                 # Результати стадій, графіки та логи метрик (JSON/CSV)
├── src/
│   ├── __init__.py
│   ├── utils.py             # Завантаження зображень, grayscale, шум, MSE/PSNR/SSIM, GLCM
│   ├── filters.py           # Ручні Гаусів фільтр, Медіанний фільтр, Собель, LBP
│   ├── segmentation.py      # Порогова бінаризація (глобальний поріг та метод Оцу)
│   └── pipelines.py         # Конвеєри обробки: Пайплайн 1 та Пайплайн 2
├── tests/
│   └── test_methods.py      # Автотести на матрицях 3x3 / 5x5 та крайніх випадках
├── main.py                  # Головний скрипт запуску через конфігурацію та CLI
├── requirements.txt         # Залежності (numpy, pillow, scipy, matplotlib)
├── STEM_Project_Report.md   # Повний дослідницький звіт за проєктом
├── META-USAGE.md            # Декларація використання інструментів ШІ
└── README.md                # Документація та інструкція запуску
```

## Встановлення та запуск

```bash
# 1. (Опціонально) Створити та активувати віртуальне середовище
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 2. Встановити залежності
pip install -r requirements.txt

# 3. Запустити повну обробку вибірки за конфігурацією
python main.py
```

### Додаткові параметри запуску (CLI)

```bash
# Запустити лише Пайплайн 1 або лише Пайплайн 2
python main.py --pipeline 1
python main.py --pipeline 2

# Перевизначити параметри методу з командного рядка без редагування коду
python main.py --pipeline 1 --filter gaussian --kernel-size 5 --sigma 1.2
python main.py --pipeline 2 --threshold-method global --threshold 127

# Запуск модульних автотестів
python -m unittest discover -s tests -v
```
