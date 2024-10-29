from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# Load the fine-tuned model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained("lora_model")

# Ensure the model is ready for inference
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Define your question
instruction = "Calculate the product of two numbers."  # e.g., "Translate the following English sentence to French."
input_text = "5 and 10"  # e.g., "Good morning"

# Format the prompt for your question
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

question_prompt = alpaca_prompt.format(instruction, input_text, "")

# Tokenize the question prompt
inputs = tokenizer([question_prompt], return_tensors="pt").to("cuda")

# Generate the response
text_streamer = TextStreamer(tokenizer)
output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)

# Print the generated response
print(output)
