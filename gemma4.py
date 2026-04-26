import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    print("=" * 60)
    print("Starting Gemma 3 Vision Batch Pipeline...")
    print("=" * 60)

    # Check if HF_TOKEN is set
    if "HF_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN environment variable is not set.")
        print("If you haven't logged in via `huggingface-cli login`, you might not be able to download Gemma 3.")
        print("You can set it inline before running: HF_TOKEN=your_token uv run python gemma_pipeline.py\n")

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
    model_id = "google/gemma-4-E4B-it"


    print(f"\nLoading Vision Processor and Model ({model_id}) in 8-bit...")
    print("This may take a moment to load from disk...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16, # Fixed deprecation warning
        quantization_config=bnb_config,
        device_map="cuda:0", # Changed from "auto" to force GPU allocation
        low_cpu_mem_usage=True
    )

    print("\nModel loaded! Starting batch processing...")
    print("-" * 60)

    results = []
    json_file_path = "response.json"
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
        #structured_prompt = f"{prompt_text}\n\nYou must output your response using EXACTLY the following structure:\n[Observation]: <your visual observations>\n[Assessment]: <your reasoning>\n[Conclusion]: <your final conclusion>"
        structured_prompt = f"{prompt_text}\n\nYou must output your response using EXACTLY the following structure:\n[Verdict]: <your answer>\n[Reasoning]: <your reasoning>"

        # Format the multimodal input
        chat_history.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": structured_prompt}
            ]
        })

        try:
            # Generate the string prompt using the chat template
            text_prompt = processor.apply_chat_template(
                chat_history,
                add_generation_prompt=True
            )

            # Extract all images from history
            images = []
            for msg in chat_history:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            images.append(item["image"])

            if not images:
                images = None

            # Pass correctly to processor
            inputs = processor(
                text=text_prompt,
                images=images,
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                #temperature=0.7,
                #top_p=0.9,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None
            )

            # Extract only the newly generated text
            input_length = inputs["input_ids"].shape[-1]
            response_ids = outputs[0][input_length:]

            response_text = processor.decode(response_ids, skip_special_tokens=True).strip()

            chat_history.append({"role": "assistant", "content": response_text})

            # Log results
            results.append({
                "file_name": filename,
                "prompt": prompt_text,
                "response": response_text
            })

            # Continuously update the JSON file so progress isn't lost if interrupted
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

            print(f"   ✅ Saved response: {response_text[:60]}...")
            
            # Free GPU memory after each image to prevent CUDA Out Of Memory!
            del inputs, outputs, response_ids
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"   ❌ Error generating response for {filename}:")
            traceback.print_exc()

            if keep_history:
                chat_history.pop()  # Remove the failed image from context
                
            # Free memory on failure too
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"Batch Processing Complete! Saved {len(results)} responses to `{json_file_path}`")

if __name__ == "__main__":
    main()