# Este é o app Streamlit que carrega o modelo de previsão de obesidade.
# Ele permite que o usuário responda perguntas sobre estilo de vida e saúde,
# e exibe uma previsão personalizada com base nos dados inseridos.

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar o modelo, scaler e label encoder
modelo = joblib.load("modelo_final.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🔍 Preditor Personalizado de Obesidade")
st.write("Responda às perguntas abaixo para receber uma previsão personalizada do seu nível de obesidade.")

# Entradas do usuário
genero = st.selectbox("Qual seu gênero?", ["Feminino", "Masculino"])
idade = st.slider("Qual sua idade?", 10, 100, 25)
altura = st.slider("Qual sua altura (em metros)?", 1.0, 2.5, 1.70)
peso = st.slider("Qual seu peso (em kg)?", 30.0, 200.0, 70.0)
historico_familiar = st.radio("Você tem histórico familiar de sobrepeso?", ["Sim", "Não"])
alimentos_caloricos = st.radio("Você consome alimentos calóricos com frequência?", ["Sim", "Não"])
vegetais = st.slider("Com que frequência consome vegetais nas refeições? (0 = nunca, 3 = sempre)", 0.0, 3.0, 2.0)
refeicoes_dia = st.slider("Quantas refeições principais faz por dia?", 1.0, 5.0, 3.0)
lanches = st.selectbox("Você costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
fuma = st.radio("Você fuma?", ["Sim", "Não"])
agua = st.slider("Quantos litros de água você bebe por dia?", 0.0, 3.0, 2.0)
controla_calorias = st.radio("Você controla a ingestão calórica?", ["Sim", "Não"])
atividade_fisica = st.slider("Horas de atividade física por semana:", 0.0, 5.0, 1.0)
tempo_tela = st.slider("Horas de uso de telas por dia (celular, TV, computador)", 0.0, 5.0, 2.0)
alcool = st.selectbox("Frequência de consumo de bebidas alcoólicas:", ["Não", "Às vezes", "Frequentemente", "Sempre"])
transporte = st.selectbox("Meio de transporte mais utilizado:", 
                          ["Transporte público", "A pé", "Carro", "Moto", "Bicicleta"])

# Montar DataFrame com as features para previsão
input_data = pd.DataFrame({
    "Age": [idade],
    "Height": [altura],
    "Weight": [peso],
    "FCVC": [vegetais],
    "NCP": [refeicoes_dia],
    "CH2O": [agua],
    "FAF": [atividade_fisica],
    "TUE": [tempo_tela],
    "Gender_Female": [1 if genero == "Feminino" else 0],
    "Gender_Male": [1 if genero == "Masculino" else 0],
    "family_history_yes": [1 if historico_familiar == "Sim" else 0],
    "family_history_no": [1 if historico_familiar == "Não" else 0],
    "FAVC_yes": [1 if alimentos_caloricos == "Sim" else 0],
    "FAVC_no": [1 if alimentos_caloricos == "Não" else 0],
    "CAEC_no": [1 if lanches == "Não" else 0],
    "CAEC_Sometimes": [1 if lanches == "Às vezes" else 0],
    "CAEC_Frequently": [1 if lanches == "Frequentemente" else 0],
    "CAEC_Always": [1 if lanches == "Sempre" else 0],
    "SMOKE_yes": [1 if fuma == "Sim" else 0],
    "SMOKE_no": [1 if fuma == "Não" else 0],
    "SCC_yes": [1 if controla_calorias == "Sim" else 0],
    "SCC_no": [1 if controla_calorias == "Não" else 0],
    "CALC_no": [1 if alcool == "Não" else 0],
    "CALC_Sometimes": [1 if alcool == "Às vezes" else 0],
    "CALC_Frequently": [1 if alcool == "Frequentemente" else 0],
    "CALC_Always": [1 if alcool == "Sempre" else 0],
    "MTRANS_Public_Transportation": [1 if transporte == "Transporte público" else 0],
    "MTRANS_Walking": [1 if transporte == "A pé" else 0],
    "MTRANS_Automobile": [1 if transporte == "Carro" else 0],
    "MTRANS_Motorbike": [1 if transporte == "Moto" else 0],
    "MTRANS_Bike": [1 if transporte == "Bicicleta" else 0]
})

# Garantir ordem correta das colunas
input_data = input_data[scaler.feature_names_in_]
input_scaled = scaler.transform(input_data)

# Explicação textual dos fatores
def gerar_explicacao():
    explicacao = []

    if vegetais < 1.0:
        explicacao.append("- Baixo consumo de vegetais")
    if alimentos_caloricos == "Sim":
        explicacao.append("- Consumo frequente de alimentos calóricos")
    if historico_familiar == "Sim":
        explicacao.append("- Histórico familiar de obesidade")
    if atividade_fisica < 1.0:
        explicacao.append("- Baixo nível de atividade física")
    if fuma == "Sim":
        explicacao.append("- Fuma atualmente")
    if alcool in ["Frequentemente", "Sempre"]:
        explicacao.append("- Consumo elevado de bebidas alcoólicas")
    if controla_calorias == "Não":
        explicacao.append("- Não controla a ingestão calórica")
    if tempo_tela > 3:
        explicacao.append("- Alto tempo de exposição a telas")

    return "Nenhum fator de risco evidente." if not explicacao else "\n".join(explicacao)

# Botão de previsão
if st.button("🔎 Prever nível de obesidade"):
    predicao = modelo.predict(input_scaled)
    resultado = label_encoder.inverse_transform(predicao)[0]

    st.success(f"✅ Resultado previsto: **{resultado.replace('_', ' ')}**")

    explicacao = gerar_explicacao()
    st.markdown("#### 🧠 Fatores que podem estar influenciando seu resultado:")
    st.markdown(f"```\n{explicacao}\n```")

    st.button("🔁 Fazer nova previsão", on_click=lambda: st.experimental_rerun())
