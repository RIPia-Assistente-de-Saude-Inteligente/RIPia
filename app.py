from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Carrega o modelo e o tokenizer na inicialização
model_name = "./hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.config.use_sliding_window_attention = False
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    messages = [
        {
            "role": "system",
            "content": (
                "Você é Qwen, um assistente útil criado pela Alibaba Cloud. "
                "Responda sempre em português. "
                "Finja ser um assistente médico, mas não prescreva medicamentos nem faça diagnósticos definitivos. "
                "Apenas sugira ações, possibilidades ou oriente a procurar um profissional de saúde."
            )
        },
        {"role": "user", "content": user_message}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)