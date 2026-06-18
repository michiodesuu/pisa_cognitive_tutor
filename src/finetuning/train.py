import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Text or VLM models using QLoRA")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--dataset_path", type=str, default="data/finetuning/textbook_reasoning.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/qlora-model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (for dry run / debugging)")
    parser.add_argument("--is_vlm", action="store_true", help="Flag to indicate if the model is a Vision-Language Model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    # QLoRA configuration (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading model: {args.model_id}")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    if args.is_vlm:
        # Load processor for VLM
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        tokenizer = processor.tokenizer
    else:
        # Load tokenizer for Text LLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=50,
        optim="paged_adamw_8bit",
        fp16=True,
        run_name="pisa_finetune",
        report_to="none" # change to "wandb" to enable wandb logging
    )
    
    # Ensure conversational format if required by SFTTrainer
    # The dataset needs to have a 'conversations' column. SFTTrainer can format it via a tokenizer template or we use format_func.
    # For standard text, we can use dataset_text_field if it was just raw text, but with conversations we use packing or chat template.
    
    # Simple formatting function using chat template
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['conversations'])):
            messages = example['conversations'][i]
            # Map 'value' to 'content' and 'from' to 'role'
            formatted_messages = [
                {"role": msg["from"], "content": msg["value"]}
                for msg in messages
            ]
            text = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving the final adapter...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
