from flask import Flask, render_template, request, jsonify, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
import pymongo
from datetime import datetime
import re

app = Flask(__name__)
app.secret_key = 'secret_key_for_session_management'

# Variáveis globais
model_ready = False
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["clinica"]
especialidades = db["especialidades"]
exames = db["exames"]
horarios = db["horarios"]
agendamentos = db["agendamentos"]

# Dados iniciais
if especialidades.count_documents({}) == 0:
    especialidades.insert_many([
        {"nome": "Clínico Geral"},
        {"nome": "Cardiologia"},
        {"nome": "Dermatologia"}
    ])

if exames.count_documents({}) == 0:
    exames.insert_many([
        {"nome": "Hemograma"},
        {"nome": "Raio-X"},
        {"nome": "Eletrocardiograma"}
    ])

if horarios.count_documents({}) == 0:
    horarios.insert_many([
        {"especialidade": "Clínico Geral", "data": "12/06/2025", "hora": "09:00", "disponivel": True},
        {"especialidade": "Clínico Geral", "data": "12/06/2025", "hora": "10:00", "disponivel": True},
        {"especialidade": "Cardiologia", "data": "13/06/2025", "hora": "14:00", "disponivel": True},
        {"especialidade": "Dermatologia", "data": "14/06/2025", "hora": "16:00", "disponivel": True}
    ])

# Funções auxiliares
def validar_data(data_texto):
    try:
        datetime.strptime(data_texto, "%d/%m/%Y")
        return True
    except ValueError:
        return False

def validar_hora(hora_texto):
    return bool(re.match(r'^\d{2}:\d{2}$', hora_texto))

def listar_opcoes(collection):
    return "\n".join(f"- {item['nome']}" for item in collection.find({}, {"_id": 0, "nome": 1}))

def process_agendamento(dados):
    disponivel = horarios.find_one({
        "especialidade": dados["especialidade"],
        "data": dados["data"],
        "hora": dados["hora"],
        "disponivel": True
    })
    if not disponivel:
        return {"erro": "Horário indisponível"}
    agendamentos.insert_one({**dados, "status": "Confirmado"})
    horarios.update_one({"_id": disponivel["_id"]}, {"$set": {"disponivel": False}})
    resumo = (
        f"Nome: {dados['nome']}\n"
        f"Telefone: {dados['telefone']}\n"
        f"E-mail: {dados['email']}\n"
        f"Especialidade: {dados['especialidade']}\n"
        f"Exame: {dados['exame'] if dados['exame'] else 'Nenhum'}\n"
        f"Data: {dados['data']}\n"
        f"Hora: {dados['hora']}"
    )
    return {"mensagem": "✅ Agendamento confirmado com sucesso!", "resumo": resumo}

# Rotas
@app.route('/')
def index():
    return render_template('chat.html' if model_ready else 'loading.html')

@app.route('/loading')
def loading_status():
    return jsonify({'model_ready': model_ready})

@app.route('/especialidades', methods=['GET'])
def listar_especialidades():
    return jsonify(list(especialidades.find({}, {"_id": 0, "nome": 1})))

@app.route('/exames', methods=['GET'])
def listar_exames():
    return jsonify(list(exames.find({}, {"_id": 0, "nome": 1})))

@app.route('/agendar', methods=['POST'])
def agendar():
    session.setdefault('step', 0)
    session.setdefault('dados', {})

    step = session['step']
    dados = session['dados']
    user_message = request.json['message'].strip()

    perguntas = [
        "Digite seu nome:",
        "Digite seu telefone:",
        "Digite seu e-mail:",
        "Digite a especialidade desejada:",
        "Digite o exame desejado (ou 'nenhum'):",
        "Digite a data do agendamento (DD/MM/AAAA):",
        "Digite o horário do agendamento (HH:MM):",
        "Confirma o agendamento? (sim/não)"
    ]

    campos = [
        "nome", "telefone", "email",
        "especialidade", "exame", "data",
        "hora", "confirmar"
    ]

    def validar_campo(campo, valor):
        if campo == "especialidade":
            if not especialidades.find_one({"nome": valor}):
                return f"Especialidade '{valor}' não encontrada.\nOpções:\n{listar_opcoes(especialidades)}"
        if campo == "exame":
            if valor.lower() != 'nenhum' and not exames.find_one({"nome": valor}):
                return f"Exame '{valor}' não encontrado.\nOpções:\n{listar_opcoes(exames)}\nOu digite 'nenhum'."
        if campo == "data":
            if not validar_data(valor):
                return "Data inválida. Informe no formato DD/MM/AAAA."
        if campo == "hora":
            if not validar_hora(valor):
                return "Horário inválido. Informe no formato HH:MM."
        if campo == "confirmar":
            if valor.lower() not in ['sim', 'não', 'nao']:
                return "Responda 'sim' para confirmar ou 'não' para cancelar."
        if campo == "telefone":
            if not re.fullmatch(r'\d{8,15}', valor):
                return "Telefone inválido. Informe apenas números, entre 8 e 15 dígitos."
        if campo == "email":
            if not re.match(r"[^@]+@[^@]+\.[^@]+", valor):
                return "E-mail inválido. Informe no formato exemplo@dominio.com."
        if campo == "nome":
            if len(valor.strip()) < 3:
                return "Nome muito curto. Informe seu nome completo."

        return None

    def obter_opcoes_para(campo):
        if campo == "especialidade":
            return f"\nOpções:\n{listar_opcoes(especialidades)}"
        if campo == "exame":
            return f"\nOpções:\n{listar_opcoes(exames)}\nOu digite 'nenhum'."
        return ""

    # === Fluxo principal ===

    if step < len(campos) - 1:  # Até antes da confirmação
        campo_atual = campos[step]

        erro = validar_campo(campo_atual, user_message)
        if erro:
            return jsonify({'response': erro})

        if campo_atual == "exame" and user_message.lower() == 'nenhum':
            dados[campo_atual] = None
        else:
            dados[campo_atual] = user_message

        session['step'] = step + 1
        session['dados'] = dados

        proxima_pergunta = perguntas[step + 1]
        opcoes = obter_opcoes_para(campos[step + 1])
        return jsonify({'response': f"{proxima_pergunta}{opcoes}"})

    elif step == len(campos) - 1:  # Etapa de confirmação
        campo_atual = campos[step]

        erro = validar_campo(campo_atual, user_message)
        if erro:
            return jsonify({'response': erro})

        if user_message.lower() in ['sim']:
            resultado = process_agendamento(dados)
            resposta = resultado.get('mensagem', 'Erro ao realizar o agendamento.')
            if 'resumo' in resultado:
                resposta += f"\nResumo do agendamento:\n{resultado['resumo']}"
        else:
            resposta = "Agendamento cancelado."

        session.pop('step', None)
        session.pop('dados', None)

        return jsonify({'response': resposta})

    else:
        return jsonify({'response': "Ocorreu um erro no fluxo de agendamento. Por favor, reinicie."})


@app.route('/chat', methods=['POST'])
def chat():
    if not model_ready:
        return jsonify({'error': 'Model is still loading!'}), 503
    user_message = request.json['message'].strip().lower()

    if user_message in ["marcar consulta", "agendar", "agendar consulta"]:
        session['step'] = 0
        session['dados'] = {}
        return jsonify({'response': "Digite seu nome:"})

    if 'step' in session:
        return agendar()

    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é RIPia, um assistente automático de saúde baseado no modelo Qwen. "
                    "Responda sempre em português, de forma clara, objetiva e educada. "
                    "Sua função é fornecer informações gerais de saúde, sem nunca prescrever medicamentos, indicar tratamentos específicos ou realizar diagnósticos. "
                    "Baseie todas as respostas apenas em informações confiáveis, amplamente reconhecidas e nunca invente dados, procedimentos ou resultados. "
                    "Se a dúvida do usuário exigir análise médica, tratamento personalizado ou diagnóstico, oriente-o a procurar um profissional de saúde qualificado. "
                    "Nunca faça afirmações categóricas ou promessas de cura. Quando não tiver certeza, diga claramente que não pode responder com precisão. "
                    "Jamais fuja do papel de assistente de saúde e nunca responda perguntas fora desse contexto."
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

# Carregamento do modelo
def load_model():
    global model, tokenizer, model_ready
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_ready = True

threading.Thread(target=load_model).start()

if __name__ == '__main__':
    app.run(debug=True)