"""
Remove rows from animal_pairs_with_ratios.csv where the Facing column is empty or null.
"""
import csv

input_file = 'animal_pairs_with_ratios.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

total = len(rows)
filtered = [row for row in rows if row.get('Facing', '').strip()]
removed = total - len(filtered)

with open(input_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered)

print(f"Total rows: {total}")
print(f"Removed rows (empty Facing): {removed}")
print(f"Remaining rows: {len(filtered)}")
