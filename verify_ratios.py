"""
Verify that generated pair images have the correct size ratios
by measuring the actual pixel dimensions of each animal in the output.
"""
import csv
import os
from PIL import Image
import re

PAIRS_CSV = "animal_pairs_with_ratios.csv"
OUTPUT_DIR = "pair_dataset"

def sanitize_name(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")

def find_animal_bboxes(img):
    """
    Find the two animals in a white-background image.
    Strategy: find non-white pixel columns, split into two groups (left animal, right animal).
    Returns (bbox1, bbox2) where each bbox is (x, y, w, h).
    """
    pixels = img.load()
    w, h = img.size
    
    # Find columns that have non-white pixels
    col_has_content = []
    for x in range(w):
        for y in range(h):
            r, g, b = pixels[x, y][:3]
            if r < 250 or g < 250 or b < 250:
                col_has_content.append(x)
                break
    
    if not col_has_content:
        return None, None
    
    # Find gap between two animals (largest gap in content columns)
    gaps = []
    for i in range(1, len(col_has_content)):
        gap = col_has_content[i] - col_has_content[i-1]
        if gap > 1:
            gaps.append((gap, i))
    
    if not gaps:
        return None, None
    
    # Largest gap separates the two animals
    gaps.sort(reverse=True)
    split_idx = gaps[0][1]
    
    left_cols = col_has_content[:split_idx]
    right_cols = col_has_content[split_idx:]
    
    def get_bbox(cols):
        x_min = min(cols)
        x_max = max(cols)
        y_min = h
        y_max = 0
        for x in range(x_min, x_max + 1):
            for y in range(h):
                r, g, b = pixels[x, y][:3]
                if r < 250 or g < 250 or b < 250:
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
                    break
            for y in range(h - 1, -1, -1):
                r, g, b = pixels[x, y][:3]
                if r < 250 or g < 250 or b < 250:
                    y_max = max(y_max, y)
                    break
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    bbox1 = get_bbox(left_cols)
    bbox2 = get_bbox(right_cols)
    return bbox1, bbox2

# Load pairs
pairs = {}
with open(PAIRS_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pair_id = row["Pair ID"].strip()
        facing_raw = row.get("Facing", "").strip()
        facing_list = [v.strip() for v in facing_raw.split(",")]
        pairs[pair_id] = {
            "animal1": row["Animal 1"].strip(),
            "animal2": row["Animal 2"].strip(),
            "length_ratio": float(row["Length Ratio (A1:A2)"]),
            "height_ratio": float(row["Height Ratio (A1:A2)"]),
            "facing_list": facing_list,
        }

# Sample a spread of pairs to verify
sample_ids = ["1", "2", "3", "10", "27", "50", "100", "150", "200", "250", "300", "350", "376"]
sample_ids = [s for s in sample_ids if s in pairs]

print(f"{'Pair':>5} | {'Animals':<45} | {'Facing':<7} | {'Expected':>8} | {'Measured':>8} | {'Error':>7}")
print("-" * 100)

errors = []
for pid in sample_ids:
    p = pairs[pid]
    for facing in p["facing_list"]:
        name1 = sanitize_name(p["animal1"])
        name2 = sanitize_name(p["animal2"])
        fname = f"pair_{pid}_{name1}_vs_{name2}_{facing}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        
        if not os.path.exists(fpath):
            print(f"  {pid:>5} | MISSING: {fname}")
            continue
        
        img = Image.open(fpath).convert("RGB")
        bbox1, bbox2 = find_animal_bboxes(img)
        
        if bbox1 is None or bbox2 is None:
            print(f"  {pid:>5} | Could not detect animals in {fname}")
            continue
        
        # Expected ratio
        if facing == "height":
            expected = p["height_ratio"]
            measured = bbox1[3] / bbox2[3]  # height1 / height2
        else:
            expected = p["length_ratio"]
            measured = bbox1[2] / bbox2[2]  # width1 / width2
        
        pct_error = abs(measured - expected) / expected * 100
        errors.append(pct_error)
        
        status = "✓" if pct_error < 5 else "✗"
        animals = f"{p['animal1']} vs {p['animal2']}"
        print(f"  {pid:>4} | {animals:<45} | {facing:<7} | {expected:>8.2f} | {measured:>8.2f} | {pct_error:>5.1f}% {status}")

if errors:
    print(f"\n{'='*50}")
    print(f"  Average error: {sum(errors)/len(errors):.2f}%")
    print(f"  Max error:     {max(errors):.2f}%")
    print(f"  Within 5%:     {sum(1 for e in errors if e < 5)}/{len(errors)}")
