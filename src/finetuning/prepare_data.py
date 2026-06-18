import os
import json
from datasets import load_dataset
from PIL import Image
from pathlib import Path

# Paths
OUTPUT_DIR = Path("data/finetuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUTPUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def prepare_textbook_reasoning(split="train", max_samples=10000):
    """
    Prepares MegaScience/TextbookReasoning dataset.
    This is text-only.
    """
    print(f"Loading MegaScience/TextbookReasoning ({split})...")
    # Using streaming or loading a small subset for demonstration
    # In reality, might want to load more
    ds = load_dataset("MegaScience/TextbookReasoning", split=split, streaming=True)
    
    formatted_data = []
    count = 0
    for item in ds:
        if count >= max_samples:
            break
        
        # TextbookReasoning typically contains 'question' and 'answer' or 'conversations'
        # Adjust mapping depending on the exact schema: usually 'instruction' / 'response' or similar
        question = item.get("question", item.get("instruction", ""))
        answer = item.get("answer", item.get("response", ""))
        
        if not question or not answer:
            continue
            
        formatted_data.append({
            "id": f"textbook_{count}",
            "conversations": [
                {"from": "user", "value": question},
                {"from": "assistant", "value": answer}
            ]
        })
        count += 1
        
    output_path = OUTPUT_DIR / "textbook_reasoning.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(formatted_data)} samples to {output_path}")

def prepare_science_qa(split="train", max_samples=10000, for_vlm=True):
    """
    Prepares derek-thomas/ScienceQA dataset.
    Has images and text.
    """
    print(f"Loading derek-thomas/ScienceQA ({split})...")
    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    
    formatted_data = []
    count = 0
    for item in ds:
        if count >= max_samples:
            break
            
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        solution = item.get("solution", "")
        hint = item.get("hint", "")
        image = item.get("image", None)
        
        # Build prompt
        prompt = f"Question: {question}\n"
        if choices:
            prompt += "Choices:\n"
            for i, c in enumerate(choices):
                prompt += f"{i}. {c}\n"
        
        if for_vlm and image is not None:
            prompt = "<image>\n" + prompt
            img_path = IMG_DIR / f"scienceqa_{count}.png"
            image.save(img_path)
            img_ref = str(img_path)
        else:
            img_ref = None
            
        # Build answer
        answer_text = ""
        if choices and 0 <= answer_idx < len(choices):
            answer_text += f"The correct answer is {answer_idx}. {choices[answer_idx]}.\n"
        if hint:
            answer_text += f"Hint: {hint}\n"
        if solution:
            answer_text += f"Explanation: {solution}\n"
            
        entry = {
            "id": f"scienceqa_{count}",
            "conversations": [
                {"from": "user", "value": prompt},
                {"from": "assistant", "value": answer_text.strip()}
            ]
        }
        if img_ref:
            entry["image"] = img_ref
            
        formatted_data.append(entry)
        count += 1
        
    suffix = "_vlm" if for_vlm else "_text"
    output_path = OUTPUT_DIR / f"scienceqa{suffix}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(formatted_data)} samples to {output_path}")

if __name__ == "__main__":
    print("Preparing datasets for Fine-Tuning...")
    # By default, load a subset to avoid huge download times during initial setup
    prepare_textbook_reasoning(max_samples=2000)
    prepare_science_qa(for_vlm=True, max_samples=2000)
    prepare_science_qa(for_vlm=False, max_samples=2000)
    print("Data preparation complete.")
