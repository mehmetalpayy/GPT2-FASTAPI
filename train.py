from helper import split_dataset, load_dataset, load_data_collator, load_model, load_tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import os
import torch

with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# print("length of dataset in characters: ", len(text))
#
# # first 1000 characters
# print(text[:1000])
#
# # all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)


class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        # lr_scheduler
        self.lr_scheduler = super().create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=self.optimizer
        )


### -------Parameters -------------
train_file_path = "input.txt"
# val_file_path = "/content/val.txt"
model_name = 'openai-community/gpt2'
output_dir = "trained_model"
overwrite_output_dir = False
per_device_train_batch_size = 8
# per_device_eval_batch_size = 8
num_train_epochs = 5
save_steps = 300
max_steps = 9000


## ----------Training-----------
def train(train_file_path,
          model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          max_steps
          ):
    tokenizer = load_tokenizer(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    model = load_model(model_name)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        # gradient_accumulation_steps=4,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=300,
        # eval_strategy="steps",
        save_steps=save_steps,
        # load_best_model_at_end=True,
        # push_to_hub=True,
        # hub_model_id="gpt2-fine-tuned",
        # hub_token=os.getenv("HF_TOKEN"),
        max_steps=max_steps,
        lr_scheduler_type="linear",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()


# Train
train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    max_steps=max_steps,
)
