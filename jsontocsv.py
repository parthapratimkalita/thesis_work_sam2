import json
import csv

with open("response_4bit.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("response.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file_name", "prompt", "response"])
    writer.writeheader()
    writer.writerows(data)

print("✅ Saved response.csv")