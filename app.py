import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load BioGPT model
MODEL_NAME = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load medical dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:1000]")

# Preprocess data for training
def preprocess_function(examples):
    tokenized_input = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=256)
    tokenized_input["labels"] = tokenized_input["input_ids"].copy()  # Labels needed for loss calculation
    return tokenized_input

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Data collator for padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",  # Fix: No eval dataset, so disable evaluation
    per_device_train_batch_size=2,
    num_train_epochs=2,
    save_strategy="no",
    report_to="none"  # Disables Hugging Face logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,  # Fix: Removed deprecated tokenizer argument
)

# Train model
trainer.train()

# Save fine-tuned model
model.save_pretrained("./fine_tuned_biogpt")
tokenizer.save_pretrained("./fine_tuned_biogpt")

# Function for chatbot response
def chatbot_response(question):
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Gradio Chat UI
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Healthcare AI Chatbot")
    
    with gr.Row():
        chatbot = gr.Chatbot(label="Medical AI Chatbot")
    
    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="Ask a medical question...")
        submit_btn = gr.Button("Send")
    
    def chat(question, chat_history):
        response = chatbot_response(question)
        chat_history.append((question, response))
        return "", chat_history

    submit_btn.click(chat, inputs=[user_input, chatbot], outputs=[user_input, chatbot])

demo.launch()
