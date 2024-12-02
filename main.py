# main.py
from dataset_loader import load_imhi_dataset
from model_loader import load_bnb_model, load_gguf_model, load_ggml_model, load_llmint8_model
from evaluate_model import evaluate_model
from memorization_loss import custom_loss_fn

# Load dataset
dataset = load_imhi_dataset()["test"]

# Select model and quantization type
model_name = "LLAMA-7B"
quantization = "4-bit"

# Load model based on quantization type
if quantization in ["2-bit", "4-bit", "8-bit"]:
    model, tokenizer = load_bnb_model(model_name, quantization)
elif quantization == "gguf":
    model = load_gguf_model(model_name)
elif quantization == "ggml":
    model = load_ggml_model(model_name)
elif quantization == "llmint.8()":
    model = load_llmint8_model(model_name)
else:
    raise ValueError("Unsupported quantization type.")

# Evaluate model
evaluation_results = evaluate_model(model, tokenizer, dataset)

# Display evaluation results
print(f"Evaluation Results for {model_name} ({quantization}):")
print(evaluation_results)
# print(executed_results)

