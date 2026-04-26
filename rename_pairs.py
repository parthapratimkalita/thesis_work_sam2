import os
import re

def rename_pairs(folder_path="pair_dataset"):
    print(f"Scanning folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"❌ Error: Directory '{folder_path}' does not exist.")
        return
        
    # Matches patterns like 'pair_045_Cheetah_vs_Axolotl_length.png'
    # Group 1 captures 'pair_045'
    # Group 2 captures '_Cheetah_vs_Axolotl_length' (the extra string we want to discard)
    # Group 3 captures '.png' or '.jpg'
    pattern = re.compile(r"^(pair_\d+)(.*)(\.[a-zA-Z0-9]+)$", re.IGNORECASE)
    
    count = 0
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            new_name = match.group(1) + match.group(3)
            
            # Only rename if it actually has extra stuff
            if filename != new_name:
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)
                
                # Check for collisions to avoid overwriting files
                if os.path.exists(new_path):
                    print(f"⚠️ Warning: '{new_name}' already exists, skipping '{filename}'")
                    continue
                    
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {filename} -> {new_name}")
                count += 1
                
    print(f"\n🎉 Successfully cleaned {count} filenames!")

if __name__ == "__main__":
    rename_pairs()
