"""
Data preparation utility for DeepLure Saree Recognition.
Supports:
1. Structuring Kaggle Indian Saree Patterns / DeepLure Drive dataset into standard layout:
   data/processed/<design_id>/<colorway_or_sample>.jpg
2. Generating a synthetic multi-colorway sanity dataset for rapid pipeline verification.
"""

import os
import argparse
from pathlib import Path
import shutil
import random
from PIL import Image, ImageDraw, ImageOps
import numpy as np


def generate_synthetic_saree_dataset(output_dir: str = "./data/synthetic_saree", num_designs: int = 15, colorways_per_design: int = 6):
    """
    Generates synthetic textile motif images across varying colorways to test the pipeline.
    Each design has a distinct spatial geometric pattern (checks, floral dots, stripes, diamond lattice),
    rendered across radically different color palettes.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Generating synthetic textile dataset with {num_designs} designs, {colorways_per_design} colorways each...")

    palettes = [
        ("crimson", "gold", "navy"),
        ("emerald", "silver", "maroon"),
        ("royal_blue", "white", "mustard"),
        ("purple", "gold", "black"),
        ("teal", "coral", "cream"),
        ("magenta", "cyan", "dark_gray"),
        ("olive", "orange", "indigo"),
        ("turquoise", "ruby", "yellow"),
    ]

    color_rgb_map = {
        "crimson": (180, 20, 50),
        "gold": (218, 165, 32),
        "navy": (10, 25, 80),
        "emerald": (20, 140, 60),
        "silver": (192, 192, 192),
        "maroon": (110, 20, 30),
        "royal_blue": (30, 60, 200),
        "white": (245, 245, 245),
        "mustard": (220, 180, 30),
        "purple": (128, 0, 128),
        "black": (25, 25, 25),
        "teal": (0, 128, 128),
        "coral": (240, 110, 90),
        "cream": (245, 240, 220),
        "magenta": (210, 30, 150),
        "cyan": (30, 200, 210),
        "dark_gray": (60, 60, 60),
        "olive": (100, 110, 30),
        "orange": (240, 130, 20),
        "indigo": (40, 10, 90),
        "turquoise": (40, 190, 180),
        "ruby": (160, 10, 40),
        "yellow": (240, 220, 30),
    }

    for d_idx in range(num_designs):
        design_name = f"design_{d_idx+1:03d}"
        design_dir = os.path.join(output_dir, design_name)
        os.makedirs(design_dir, exist_ok=True)

        # Pick distinct geometric pattern rule for this design ID
        pattern_type = d_idx % 5

        for c_idx in range(colorways_per_design):
            palette = palettes[(d_idx + c_idx) % len(palettes)]
            bg_color = color_rgb_map[palette[0]]
            motif_color = color_rgb_map[palette[1]]
            accent_color = color_rgb_map[palette[2]]

            img = Image.new("RGB", (256, 256), color=bg_color)
            draw = ImageDraw.Draw(img)

            # Draw distinctive motifs
            if pattern_type == 0:  # Diamond lattice motif (Zari checks)
                for x in range(0, 256, 32):
                    draw.line([(x, 0), (x + 128, 256)], fill=motif_color, width=2)
                    draw.line([(x, 256), (x + 128, 0)], fill=motif_color, width=2)
                # Saree Border
                draw.rectangle([(0, 220), (256, 256)], fill=accent_color)

            elif pattern_type == 1:  # Polka dots / Butis motif
                for x in range(20, 256, 40):
                    for y in range(20, 220, 40):
                        draw.ellipse([(x - 8, y - 8), (x + 8, y + 8)], fill=motif_color)
                        draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=accent_color)
                # Temple border
                for x in range(0, 256, 20):
                    draw.polygon([(x, 256), (x + 10, 230), (x + 20, 256)], fill=accent_color)

            elif pattern_type == 2:  # Vertical pinstripes / Leheriya chevron
                for x in range(0, 256, 24):
                    draw.line([(x, 0), (x, 220)], fill=motif_color, width=4)
                    draw.line([(x + 8, 0), (x + 8, 220)], fill=accent_color, width=2)
                draw.rectangle([(0, 220), (256, 256)], fill=accent_color)

            elif pattern_type == 3:  # Paisley / Floral Jaal mesh
                for x in range(30, 256, 60):
                    for y in range(30, 220, 60):
                        draw.arc([(x - 15, y - 20), (x + 15, y + 20)], start=0, end=270, fill=motif_color, width=3)
                        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=accent_color)
                draw.rectangle([(0, 210), (256, 256)], fill=accent_color)

            else:  # Concentric geometric mandalas
                for x in [64, 192]:
                    for y in [64, 160]:
                        draw.ellipse([(x - 30, y - 30), (x + 30, y + 30)], outline=motif_color, width=3)
                        draw.ellipse([(x - 15, y - 15), (x + 15, y + 15)], fill=accent_color)
                draw.rectangle([(0, 220), (256, 256)], fill=accent_color)

            # Add subtle natural textile weave noise
            noise = np.random.randint(-10, 10, (256, 256, 3), dtype=np.int16)
            img_arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            final_img = Image.fromarray(img_arr)

            filename = f"colorway_{c_idx+1:02d}_{palette[0]}_{palette[1]}.jpg"
            final_img.save(os.path.join(design_dir, filename), quality=95)

    print(f"[+] Synthetic dataset successfully generated at: {output_dir}")


def organize_raw_dataset(raw_dir: str, target_dir: str):
    """
    Standardizes folders from Kaggle or external sources into target_dir/<design_id>/<image>.jpg
    """
    os.makedirs(target_dir, exist_ok=True)
    raw_path = Path(raw_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # If raw dir already has subfolders
    subdirs = [d for d in raw_path.iterdir() if d.is_dir()]
    if subdirs:
        for d in subdirs:
            out_sub = Path(target_dir) / d.name
            out_sub.mkdir(exist_ok=True)
            for f in d.iterdir():
                if f.suffix.lower() in image_exts:
                    shutil.copy(f, out_sub / f.name)
        print(f"[+] Re-organized folder structured data into {target_dir}")
    else:
        print(f"[-] No subdirectories found in {raw_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for Saree Recognition")
    parser.add_argument("--raw_dir", type=str, default=None, help="Path to raw downloaded dataset")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic multi-colorway dataset")
    parser.add_argument("--num_designs", type=int, default=20, help="Number of synthetic designs")
    parser.add_argument("--colorways", type=int, default=6, help="Colorways per synthetic design")

    args = parser.parse_args()

    if args.synthetic or args.raw_dir is None:
        generate_synthetic_saree_dataset(
            output_dir=args.output_dir,
            num_designs=args.num_designs,
            colorways_per_design=args.colorways,
        )
    else:
        organize_raw_dataset(args.raw_dir, args.output_dir)
