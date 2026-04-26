import os
import json
import csv
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

def main():
    print("=" * 60)
    print("Starting Qwen 2.5 VL Vision Batch Pipeline...")
    print("=" * 60)

    # 1. Ask User for Setup Preferences
    folder_path = input("\nEnter the folder path containing images [default: pair_dataset]: ").strip()
    if not folder_path: folder_path = "pair_dataset"

    prompt_text = input('Enter your question [default: "Are the objects in their correct relative sizes?"]: ').strip()
    if not prompt_text: prompt_text = "Are the objects in their correct relative sizes?"

    keep_history = input("Should we keep past history between images? (yes/no) [default: no]: ").strip().lower()
    keep_history = keep_history == "yes"

    if not os.path.exists(folder_path):
        print(f"\n❌ Error: The folder '{folder_path}' does not exist. Please check your path.")
        return

    # 2. Load Model & Processor
    # Use 7B model. If you get OOM errors, change to "Qwen/Qwen2.5-VL-3B-Instruct"
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    print(f"\nLoading Vision Processor and Model ({model_id}) in 4-bit (bfloat16)...")
    print("Optimized for 8 GB RTX 4060...")
    print("This may take a moment to load from disk...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("\nModel loaded! Starting batch processing...")
    print("-" * 60)

    results = []
    json_file_path = "response.json"
    csv_file_path = "response.csv"
    chat_history = []

    # 3. Iterate over Images
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    if not image_files:
        print(f"No images found in {folder_path}!")
        return

    for index, filename in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[{index}/{len(image_files)}] ❌ Skipping Corrupted Image {filename}: {e}")
            continue

        print(f"[{index}/{len(image_files)}] ⏳ Processing {filename}...")

        if not keep_history:
            chat_history = []

        # Wrap the Prompt to enforce structure
        structured_prompt = f"{prompt_text}\n\nYou must output your response using EXACTLY the following structure:\n[Verdict]: <your answer>\n[Reasoning]: <your reasoning>"

        chat_history.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": structured_prompt}
            ]
        })

        try:
            text_prompt = processor.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(chat_history)

            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )

            input_length = inputs["input_ids"].shape[-1]
            response_ids = outputs[0][input_length:]

            response_text = processor.decode(response_ids, skip_special_tokens=True).strip()

            chat_history.append({"role": "assistant", "content": response_text})

            results.append({
                "file_name": filename,
                "prompt": prompt_text,
                "response": response_text
            })

            # Save JSON (continuously)
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

            # Save CSV (continuously)
            #with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
            #    writer = csv.DictWriter(f, fieldnames=["file_name", "prompt", "response"])
            #    writer.writeheader()
            #    writer.writerows(results)

            print(f"   ✅ Saved response: {response_text[:60]}...")

            # Free GPU memory after each image
            del inputs, outputs, response_ids
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"   ❌ Error generating response for {filename}:")
            traceback.print_exc()

            if keep_history:
                chat_history.pop()

            # Free GPU memory on error too
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"Batch Processing Complete! Saved {len(results)} responses to `{json_file_path}` and `{csv_file_path}`")

if __name__ == "__main__":
    main()