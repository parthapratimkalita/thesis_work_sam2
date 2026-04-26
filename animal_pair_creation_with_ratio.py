import pandas as pd
from itertools import combinations

# ── 1. Read source files ────────────────────────────────────────────────
img_info = pd.read_csv("img_file_info.csv")
size_info = pd.read_csv("data_size_info.csv")

# ── 2. Parse the 'facing' column into proper lists ─────────────────────
#       Handle: NaN, empty strings, whitespace-only, "nan" as string
def parse_facing(val):
    if pd.isna(val):                        # actual NaN / None
        return []
    val = str(val).strip()
    if val == "" or val.lower() == "nan":   # empty or stringified NaN
        return []
    items = [item.strip() for item in val.split(",")]
    # Remove any remaining empty strings or whitespace-only items
    items = [item for item in items if item]
    return items

img_info["facing_list"] = img_info["facing"].apply(parse_facing)

# ── 3. Filter out animals whose facing list is empty ───────────────────
valid_animals = img_info[img_info["facing_list"].apply(len) > 0].copy()

print(f"Total animals         : {len(img_info)}")
print(f"Valid animals (facing) : {len(valid_animals)}")

# Log any dropped animals
dropped = img_info[img_info["facing_list"].apply(len) == 0]
if not dropped.empty:
    print("⚠ Dropped animals with empty/null facing:")
    for _, row in dropped.iterrows():
        print(f"   - {row['Animal']}  (facing raw: {repr(row['facing'])})")

# ── 4. Build lookup dictionaries ───────────────────────────────────────
animal_facing = dict(zip(valid_animals["Animal"], valid_animals["facing_list"]))

size_lookup = {}
for _, row in size_info.iterrows():
    size_lookup[row["Animal"]] = {
        "height": row["Total Height (m)"],
        "length": row["Length (m)"],
    }

# ── 5. Generate every unique pair and compute common facing + ratio ────
animals = valid_animals["Animal"].tolist()
pairs = list(combinations(animals, 2))

results = []
skipped_no_common = 0

for a1, a2 in pairs:
    list1 = animal_facing[a1]
    list2 = animal_facing[a2]

    # Find common facing values (preserve order from animal 1's list)
    common = [f for f in list1 if f in list2]

    # Skip the pair if they share no common facing direction
    if not common:
        skipped_no_common += 1
        continue

    # If two matches exist, keep only the FIRST one
    chosen_facing = common[0]

    # Look up the relevant dimension for each animal
    if chosen_facing == "height":
        size_a1 = size_lookup[a1]["height"]
        size_a2 = size_lookup[a2]["height"]
    else:  # "length"
        size_a1 = size_lookup[a1]["length"]
        size_a2 = size_lookup[a2]["length"]

    # Extra safety: skip if either size is NaN or zero
    if pd.isna(size_a1) or pd.isna(size_a2) or size_a2 == 0:
        continue

    ratio = round(size_a1 / size_a2, 4)

    results.append(
        {
            "Animal 1": a1,
            "Animal 2": a2,
            "Facing": chosen_facing,
            "Size Animal 1 (m)": size_a1,
            "Size Animal 2 (m)": size_a2,
            "Ratio (A1/A2)": ratio,
        }
    )

# ── 6. Save to CSV ─────────────────────────────────────────────────────
result_df = pd.DataFrame(results)
result_df.to_csv("animal_pairs.csv", index=False)

print(f"\nSkipped (no common)   : {skipped_no_common}")
print(f"Total pairs generated : {len(results)}")
print(f"Output saved to       : animal_pairs.csv")
print("\nFirst 10 rows:")
print(result_df.head(10).to_string(index=False))