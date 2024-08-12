from transformers import TextDataset, DataCollatorForLanguageModeling
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def split_dataset(file_path, train_size=0.9):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Calculate split index
    split_idx = int(len(text) * train_size)

    # Split text into train and validation sets
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Save the train and validation texts
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.write(train_text)

    with open('val.txt', 'w', encoding='utf-8') as f:
        f.write(val_text)

    return 'train.txt', 'val.txt'


def load_dataset(file_path, tokenizer, block_size=64):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def llm_output(model, tokenizer, input_text):

    try:
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        final_outputs = model.generate(
            inputs,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            max_length=100,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Error in generate_text: {e}")
        raise
