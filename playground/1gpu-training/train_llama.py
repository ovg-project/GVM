from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, Trainer, TrainingArguments
from trl import SFTTrainer


def prepare_dataset(dataset_path: str, model_path: str):
    from datasets import load_dataset

    def tokenize(example):
        # tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer(example['text'], truncation=True, max_length=512)

    dataset = load_dataset(dataset_path, split="train")
    tokenized = dataset.map(tokenize, batched=True)

    return tokenized


def train(dataset_path: str, model_path: str):
    from datasets import load_dataset
    # tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # tokenized = prepare_dataset(dataset_path=dataset_path,
    #                             model_path=model_path)
    train_dataset = load_dataset(dataset_path, split="train")

    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")

    training_args = TrainingArguments(
        output_dir="./llama3b-finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_steps=500,
        logging_steps=50,
        num_train_epochs=3,
        fp16=True,
        optim="adamw_torch_fused",
        save_total_limit=2,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        packing=False,
    )

    trainer.train()


def replace_default_allocator():
    """
    Replace the default CUDA allocator with a custom one.
    """
    import os
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LIB_PATH = os.path.join(ROOT_DIR, "csrc/custom_allocator/uvm_allocator.so")
    # Set the custom allocator
    from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator
    change_current_allocator(
        CUDAPluggableAllocator(LIB_PATH, "uvm_alloc", "uvm_free"))
    print("Default allocator replaced with custom UVM allocator.")


if __name__ == "__main__":
    replace_default_allocator()
    dataset_path = "mlabonne/guanaco-llama2-1k"
    model_path = "meta-llama/Llama-3.2-3B"
    train(dataset_path=dataset_path, model_path=model_path)
