"""
Pipeline to generate dataset images with pairs of animals on a white background,
scaled so that their size ratio matches the ratio from animal_pairs.csv.

For each pair:
  - If Facing = "height"  → scale so (Animal1 image height) / (Animal2 image height) = Ratio
  - If Facing = "length"  → scale so (Animal1 image width)  / (Animal2 image width)  = Ratio

Both animals are composited onto a white canvas, placed side by side.
"""

import csv
import os
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────
EXTRACTED_DIR = "extracted"
OUTPUT_DIR = "pair_dataset"
IMG_FILE_INFO = "img_file_info.csv"
PAIRS_CSV = "animal_pairs.csv"

CANVAS_WIDTH = 2000
CANVAS_HEIGHT = 1200
PADDING = 40
MIN_ANIMAL_PX = 30
MAX_ANIMAL_PX = 1100


def load_animal_filenames(csv_path):
    """Load animal name → image filename mapping."""
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            animal = row["Animal"].strip()
            filename = row["Filename"].strip()
            mapping[animal] = filename
    return mapping


def load_pairs(csv_path):
    """
    Load animal pairs from animal_pairs.csv.
    Expected columns:
        Animal 1, Animal 2, Facing, Size Animal 1 (m), Size Animal 2 (m), Ratio (A1/A2)
    """
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            facing = row.get("Facing", "").strip()
            if not facing:
                continue

            ratio_str = row.get("Ratio (A1/A2)", "").strip()
            if not ratio_str:
                continue

            try:
                ratio = float(ratio_str)
            except ValueError:
                print(f"  WARNING: Invalid ratio '{ratio_str}' at row {idx}, skipping.")
                continue

            pairs.append({
                "pair_id": idx,
                "animal1": row["Animal 1"].strip(),
                "animal2": row["Animal 2"].strip(),
                "facing": facing,
                "size_a1": float(row["Size Animal 1 (m)"]),
                "size_a2": float(row["Size Animal 2 (m)"]),
                "ratio": ratio,
            })
    return pairs


def get_tight_bbox(img):
    """Get the bounding box of non-transparent pixels (for RGBA images)."""
    if img.mode == "RGBA":
        alpha = img.split()[3]
        bbox = alpha.getbbox()
        if bbox:
            return bbox
    return (0, 0, img.width, img.height)


def crop_to_content(img):
    """Crop image to its non-transparent content."""
    bbox = get_tight_bbox(img)
    if bbox:
        return img.crop(bbox)
    return img


def scale_animal_images(img1, img2, ratio, facing):
    """
    Scale two animal images so their size ratio on the relevant axis matches `ratio`.

    facing = "height" → ratio = img1_height / img2_height
    facing = "length" → ratio = img1_width  / img2_width

    Returns (scaled_img1, scaled_img2).
    """
    # Crop to content first to get accurate dimensions
    img1 = crop_to_content(img1)
    img2 = crop_to_content(img2)

    w1, h1 = img1.size
    w2, h2 = img2.size

    if facing == "height":
        # We want: new_h1 / new_h2 = ratio
        if ratio >= 1:
            # Animal 1 is taller (or equal)
            ref_h1 = min(MAX_ANIMAL_PX, CANVAS_HEIGHT - 2 * PADDING)
            ref_h2 = ref_h1 / ratio
        else:
            # Animal 2 is taller
            ref_h2 = min(MAX_ANIMAL_PX, CANVAS_HEIGHT - 2 * PADDING)
            ref_h1 = ref_h2 * ratio

        # Ensure minimum size
        if ref_h1 < MIN_ANIMAL_PX:
            ref_h1 = MIN_ANIMAL_PX
            ref_h2 = ref_h1 / ratio
        if ref_h2 < MIN_ANIMAL_PX:
            ref_h2 = MIN_ANIMAL_PX
            ref_h1 = ref_h2 * ratio

        # Scale factors (maintain aspect ratio per animal)
        scale1 = ref_h1 / h1
        scale2 = ref_h2 / h2

    elif facing == "length":
        # We want: new_w1 / new_w2 = ratio
        available_width = CANVAS_WIDTH - 3 * PADDING

        if ratio >= 1:
            # Animal 1 is longer
            # new_w1 + new_w2 <= available_width
            # new_w1 = ratio * new_w2
            ref_w2 = available_width / (ratio + 1)
            ref_w1 = ref_w2 * ratio
        else:
            ref_w1 = available_width / (1 + 1 / ratio)
            ref_w2 = ref_w1 / ratio

        # Cap individual widths
        max_w = min(MAX_ANIMAL_PX, available_width * 0.8)
        if ref_w1 > max_w:
            ref_w1 = max_w
            ref_w2 = ref_w1 / ratio
        if ref_w2 > max_w:
            ref_w2 = max_w
            ref_w1 = ref_w2 * ratio

        # Ensure minimum size
        if ref_w1 < MIN_ANIMAL_PX:
            ref_w1 = MIN_ANIMAL_PX
            ref_w2 = ref_w1 / ratio
        if ref_w2 < MIN_ANIMAL_PX:
            ref_w2 = MIN_ANIMAL_PX
            ref_w1 = ref_w2 * ratio

        scale1 = ref_w1 / w1
        scale2 = ref_w2 / w2
    else:
        raise ValueError(f"Unknown facing direction: '{facing}'")

    # Resize (maintain aspect ratio using the computed scale factor)
    new_w1 = max(1, int(w1 * scale1))
    new_h1 = max(1, int(h1 * scale1))
    new_w2 = max(1, int(w2 * scale2))
    new_h2 = max(1, int(h2 * scale2))

    # Ensure they fit on canvas vertically
    max_canvas_h = CANVAS_HEIGHT - 2 * PADDING
    if new_h1 > max_canvas_h:
        downscale = max_canvas_h / new_h1
        new_w1 = max(1, int(new_w1 * downscale))
        new_h1 = max(1, int(new_h1 * downscale))
        new_w2 = max(1, int(new_w2 * downscale))
        new_h2 = max(1, int(new_h2 * downscale))
    if new_h2 > max_canvas_h:
        downscale = max_canvas_h / new_h2
        new_w1 = max(1, int(new_w1 * downscale))
        new_h1 = max(1, int(new_h1 * downscale))
        new_w2 = max(1, int(new_w2 * downscale))
        new_h2 = max(1, int(new_h2 * downscale))

    # Ensure total width fits
    total_w = new_w1 + new_w2 + 3 * PADDING
    if total_w > CANVAS_WIDTH:
        downscale = (CANVAS_WIDTH - 3 * PADDING) / (new_w1 + new_w2)
        new_w1 = max(1, int(new_w1 * downscale))
        new_h1 = max(1, int(new_h1 * downscale))
        new_w2 = max(1, int(new_w2 * downscale))
        new_h2 = max(1, int(new_h2 * downscale))

    scaled1 = img1.resize((new_w1, new_h1), Image.LANCZOS)
    scaled2 = img2.resize((new_w2, new_h2), Image.LANCZOS)

    return scaled1, scaled2


def verify_ratio(scaled1, scaled2, target_ratio, facing):
    """Verify the achieved ratio matches the target (for logging)."""
    w1, h1 = scaled1.size
    w2, h2 = scaled2.size

    if facing == "height":
        achieved = h1 / h2 if h2 > 0 else 0
    else:
        achieved = w1 / w2 if w2 > 0 else 0

    error_pct = abs(achieved - target_ratio) / target_ratio * 100 if target_ratio > 0 else 0
    return achieved, error_pct


def compose_pair_image(scaled1, scaled2):
    """
    Compose two scaled animal images onto a white canvas.
    Animals are placed side by side, bottom-aligned.
    """
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255))

    w1, h1 = scaled1.size
    w2, h2 = scaled2.size

    total_w = w1 + w2 + PADDING
    start_x = (CANVAS_WIDTH - total_w) // 2

    # Bottom-align both animals
    y1 = CANVAS_HEIGHT - PADDING - h1
    y2 = CANVAS_HEIGHT - PADDING - h2

    canvas.paste(scaled1, (start_x, y1), scaled1 if scaled1.mode == "RGBA" else None)
    canvas.paste(
        scaled2,
        (start_x + w1 + PADDING, y2),
        scaled2 if scaled2.mode == "RGBA" else None,
    )

    return canvas


def sanitize_name(name):
    """Make a filesystem-safe name."""
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    animal_files = load_animal_filenames(IMG_FILE_INFO)
    pairs = load_pairs(PAIRS_CSV)

    print(f"Loaded {len(animal_files)} animal image mappings")
    print(f"Loaded {len(pairs)} pairs from {PAIRS_CSV}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print()

    # ── Pre-load all animal images ─────────────────────────────────────
    image_cache = {}
    for animal, filename in animal_files.items():
        path = os.path.join(EXTRACTED_DIR, filename)
        if os.path.exists(path):
            image_cache[animal] = Image.open(path).convert("RGBA")
        else:
            print(f"  WARNING: Image not found for '{animal}': {path}")

    generated = 0
    skipped = 0
    ratio_errors = []

    for pair in pairs:
        pid = pair["pair_id"]
        a1 = pair["animal1"]
        a2 = pair["animal2"]
        facing = pair["facing"]
        target_ratio = pair["ratio"]

        # ── Validate ───────────────────────────────────────────────────
        if a1 not in image_cache:
            print(f"  SKIP pair {pid}: no image for '{a1}'")
            skipped += 1
            continue
        if a2 not in image_cache:
            print(f"  SKIP pair {pid}: no image for '{a2}'")
            skipped += 1
            continue
        if facing not in ("height", "length"):
            print(f"  SKIP pair {pid}: unknown facing '{facing}'")
            skipped += 1
            continue
        if target_ratio <= 0:
            print(f"  SKIP pair {pid}: invalid ratio {target_ratio}")
            skipped += 1
            continue

        # ── Scale & compose ────────────────────────────────────────────
        img1 = image_cache[a1].copy()
        img2 = image_cache[a2].copy()

        try:
            scaled1, scaled2 = scale_animal_images(img1, img2, target_ratio, facing)

            # Verify the achieved ratio
            achieved, error_pct = verify_ratio(scaled1, scaled2, target_ratio, facing)
            if error_pct > 5:
                ratio_errors.append(
                    f"  Pair {pid} ({a1} vs {a2}, {facing}): "
                    f"target={target_ratio:.4f}, achieved={achieved:.4f}, "
                    f"error={error_pct:.1f}%"
                )

            canvas = compose_pair_image(scaled1, scaled2)

            # ── Save ───────────────────────────────────────────────────
            name1 = sanitize_name(a1)
            name2 = sanitize_name(a2)
            out_name = f"pair_{pid:03d}_{name1}_vs_{name2}_{facing}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            canvas.save(out_path, "PNG")
            generated += 1

            if generated % 50 == 0:
                print(f"  Generated {generated} images...")

        except Exception as e:
            print(f"  ERROR pair {pid} ({a1} vs {a2}, {facing}): {e}")
            skipped += 1

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Generated : {generated} images")
    print(f"  Skipped   : {skipped}")
    print(f"  Output    : {OUTPUT_DIR}/")

    if ratio_errors:
        print(f"\n  ⚠ Ratio accuracy warnings ({len(ratio_errors)}):")
        for msg in ratio_errors:
            print(msg)
    else:
        print(f"  ✓ All ratios within 5% tolerance")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()