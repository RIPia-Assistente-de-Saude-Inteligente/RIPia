from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading

app = Flask(__name__)

# Variável para verificar o status de carregamento do modelo
model_ready = False


# Função para carregar o modelo e tokenizer em um thread separado
def load_model():
    global model, tokenizer, model_ready
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.use_sliding_window_attention = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_ready = True


# Inicializa o carregamento do modelo em um thread separado
threading.Thread(target=load_model).start()


@app.route('/')
def index():
    if model_ready:
        return render_template('chat.html')
    else:
        return render_template('loading.html')  # Carrega a tela de carregamento


@app.route('/loading')
def loading_status():
    return jsonify({'model_ready': model_ready})


@app.route('/chat', methods=['POST'])
def chat():
    if not model_ready:
        return jsonify({'error': 'Model is still loading!'}), 503

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
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)