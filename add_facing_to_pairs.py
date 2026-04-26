"""
Add facing information to animal_pairs_with_ratios.csv by intersecting
the facing values of both animals in each pair from img_file_info.csv.
"""
import csv

# Step 1: Load facing info from img_file_info.csv
animal_facing = {}
with open('img_file_info.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        animal = row['Animal'].strip()
        facing_raw = row['facing'].strip()
        # Parse into a set of individual facing values
        facing_set = set(v.strip() for v in facing_raw.split(','))
        animal_facing[animal] = facing_set

print("=== Animal Facing Map ===")
for animal, facing in sorted(animal_facing.items()):
    print(f"  {animal}: {facing}")

# Step 2: Read animal_pairs_with_ratios.csv and add facing column
rows_out = []
no_match_pairs = []

with open('animal_pairs_with_ratios.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ['Facing']
    
    for row in reader:
        animal1 = row['Animal 1'].strip()
        animal2 = row['Animal 2'].strip()
        
        facing1 = animal_facing.get(animal1, set())
        facing2 = animal_facing.get(animal2, set())
        
        if not facing1:
            print(f"WARNING: No facing info for '{animal1}'")
        if not facing2:
            print(f"WARNING: No facing info for '{animal2}'")
        
        # Intersection of facing values
        common_facing = facing1 & facing2
        
        if common_facing:
            # Sort for consistent output
            facing_str = ', '.join(sorted(common_facing))
        else:
            facing_str = ''
            no_match_pairs.append((row['Pair ID'], animal1, animal2, facing1, facing2))
        
        row['Facing'] = facing_str
        rows_out.append(row)

# Step 3: Write output
with open('animal_pairs_with_ratios.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print(f"\n=== Done ===")
print(f"Total pairs processed: {len(rows_out)}")
print(f"Pairs with no common facing: {len(no_match_pairs)}")
if no_match_pairs:
    print("\nPairs with no common facing:")
    for pair_id, a1, a2, f1, f2 in no_match_pairs:
        print(f"  Pair {pair_id}: {a1} ({f1}) vs {a2} ({f2})")
