"""
Подготовка датасета "кошки vs собаки" для train.py
-----------------------------------------------------
Скачивает открытый датасет с Kaggle (через kagglehub) и раскладывает
случайную подвыборку изображений в структуру, ожидаемую train.py:

    data/
        train/
            cat/
            dog/
        val/
            cat/
            dog/

Использование:
    pip install kagglehub
    python prepare_dataset.py --per-class 50 --val-split 0.2

Примечание: для скачивания с Kaggle нужен аккаунт kaggle.com и
API-токен (kaggle.json) — kagglehub попросит авторизоваться при первом
запуске, либо выполните `kaggle` login заранее.
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Подготовка датасета cat/dog")
    parser.add_argument("--per-class", type=int, default=50,
                         help="Сколько изображений на класс взять всего (train+val)")
    parser.add_argument("--val-split", type=float, default=0.2,
                         help="Доля изображений на класс, уходящая в val")
    parser.add_argument("--output", type=str, default="./data",
                         help="Куда разложить итоговую структуру train/val")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def download_source_dataset() -> Path:
    """Скачивает 'Cat and Dog' датасет с Kaggle через kagglehub и возвращает путь."""
    import kagglehub
    path = kagglehub.dataset_download("tongpython/cat-and-dog")
    return Path(path)


def find_images_by_class(source_root: Path):
    """
    Датасет tongpython/cat-and-dog содержит папки вида:
        training_set/training_set/cats/*.jpg
        training_set/training_set/dogs/*.jpg
    Собираем все пути картинок по каждому классу, независимо от вложенности.
    """
    cats, dogs = [], []
    for img_path in source_root.rglob("*.jpg"):
        parent = img_path.parent.name.lower()
        if "cat" in parent:
            cats.append(img_path)
        elif "dog" in parent:
            dogs.append(img_path)
    return {"cat": cats, "dog": dogs}


def build_split(images_by_class: dict, per_class: int, val_split: int, seed: int):
    random.seed(seed)
    split = {}
    for cls, images in images_by_class.items():
        if len(images) < per_class:
            print(f"[!] Для класса '{cls}' найдено только {len(images)} изображений "
                  f"(запрошено {per_class}) — берём все доступные")
        sample = random.sample(images, min(per_class, len(images)))
        n_val = max(1, int(len(sample) * val_split))
        split[cls] = {"val": sample[:n_val], "train": sample[n_val:]}
    return split


def copy_split(split: dict, output_dir: Path):
    for cls, parts in split.items():
        for part_name, files in parts.items():
            dest_dir = output_dir / part_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dest_dir / src.name)
            print(f"[*] {part_name}/{cls}: {len(files)} изображений")


def main():
    args = parse_args()
    output_dir = Path(args.output)

    print("[*] Скачивание датасета с Kaggle (может занять пару минут)...")
    source_root = download_source_dataset()
    print(f"[*] Датасет скачан в: {source_root}")

    images_by_class = find_images_by_class(source_root)
    print(f"[*] Найдено: cat={len(images_by_class['cat'])}, dog={len(images_by_class['dog'])}")

    split = build_split(images_by_class, args.per_class, args.val_split, args.seed)
    copy_split(split, output_dir)

    print(f"\n[*] Готово. Данные разложены в: {output_dir.resolve()}")
    print("[*] Можно запускать: python train.py --data-dir ./data --epochs 15 --freeze-backbone")


if __name__ == "__main__":
    main()
