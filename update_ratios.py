"""
Update ratios in animal_pairs_with_ratios.csv using actual dimensions
from data_size_info.csv.

Ratio = Animal1 dimension / Animal2 dimension (rounded to 2 decimal places).
"""
import csv

SIZE_INFO = "data_size_info.csv"
PAIRS_CSV = "animal_pairs_with_ratios.csv"

# Step 1: Load animal dimensions
animals = {}
with open(SIZE_INFO, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["Animal"].strip()
        animals[name] = {
            "length": float(row["Length (m)"]),
            "height": float(row["Total Height (m)"]),
            "width": float(row["Width (m)"]),
        }

print(f"Loaded dimensions for {len(animals)} animals")

# Step 2: Read pairs and recalculate ratios
rows_out = []
updated = 0
with open(PAIRS_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        a1 = row["Animal 1"].strip()
        a2 = row["Animal 2"].strip()

        if a1 in animals and a2 in animals:
            old_lr = row["Length Ratio (A1:A2)"]
            old_hr = row["Height Ratio (A1:A2)"]
            old_wr = row["Width Ratio (A1:A2)"]

            new_lr = round(animals[a1]["length"] / animals[a2]["length"], 2)
            new_hr = round(animals[a1]["height"] / animals[a2]["height"], 2)
            new_wr = round(animals[a1]["width"] / animals[a2]["width"], 2)

            row["Length Ratio (A1:A2)"] = str(new_lr)
            row["Height Ratio (A1:A2)"] = str(new_hr)
            row["Width Ratio (A1:A2)"] = str(new_wr)

            if old_lr != str(new_lr) or old_hr != str(new_hr) or old_wr != str(new_wr):
                updated += 1
                print(f"  Pair {row['Pair ID']}: {a1} vs {a2}")
                print(f"    Length: {old_lr} -> {new_lr}")
                print(f"    Height: {old_hr} -> {new_hr}")
                print(f"    Width:  {old_wr} -> {new_wr}")
        else:
            if a1 not in animals:
                print(f"WARNING: '{a1}' not found in size data")
            if a2 not in animals:
                print(f"WARNING: '{a2}' not found in size data")

        rows_out.append(row)

# Step 3: Write back
with open(PAIRS_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print(f"\nDone! Updated {updated} out of {len(rows_out)} pairs.")
