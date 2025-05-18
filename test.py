from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "./hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

# Load model with explicitly disabled sliding window attention
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"
)
model.config.use_sliding_window_attention = False  # Disable sliding window attention

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare prompt and messages
prompt = "Me dê uma pequena introdução sobre LLMs."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant who can answer questions in portuguese."},
    {"role": "user", "content": prompt}
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Encode inputs
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate output
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# Extract response
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print response
print(response)
