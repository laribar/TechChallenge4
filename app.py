# Este é o app Streamlit que irá carregar o modelo salvo, coletar os dados do usuário,
# aplicar as mesmas transformações do pipeline (normalização e codificação)
# e retornar a predição do nível de obesidade com uma interface simples e intuitiva.

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar modelo, scaler e label encoder
modelo = joblib.load("modelo_obesidade.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🔍 Preditor de Obesidade")
st.write("Preencha os dados abaixo para prever o nível de obesidade:")

# Inputs do usuário
genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
idade = st.slider("Idade", 10, 100, 25)
altura = st.slider("Altura (em metros)", 1.0, 2.5, 1.70)
peso = st.slider("Peso (em kg)", 30.0, 200.0, 70.0)
historico_familiar = st.selectbox("Algum familiar sofre ou sofreu com sobrepeso?", ["Sim", "Não"])
alimentos_caloricos = st.selectbox("Você consome alimentos muito calóricos com frequência?", ["Sim", "Não"])
vegetais = st.slider("Você costuma comer vegetais nas refeições? (0 = nunca, 3 = sempre)", 0.0, 3.0, 2.0)
refeicoes_dia = st.slider("Quantas refeições principais você faz por dia?", 1.0, 5.0, 3.0)
lanches = st.selectbox("Você costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
fuma = st.selectbox("Você fuma?", ["Sim", "Não"])
agua = st.slider("Quantos litros de água você bebe por dia?", 0.0, 3.0, 2.0)
controla_calorias = st.selectbox("Você controla a quantidade de calorias que consome?", ["Sim", "Não"])
atividade_fisica = st.slider("Quantas horas de atividade física você pratica por semana?", 0.0, 5.0, 1.0)
tempo_tela = st.slider("Tempo de uso de dispositivos tecnológicos por dia (em horas)", 0.0, 5.0, 2.0)
alcool = st.selectbox("Com que frequência você consome bebida alcoólica?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
transporte = st.selectbox("Qual meio de transporte você mais utiliza?", 
                          ["Transporte público", "A pé", "Carro", "Moto", "Bicicleta"])

# Conversão para os valores esperados pelo modelo
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

# Normalizar os dados
input_scaled = scaler.transform(input_data)

# Fazer a predição
if st.button("Prever"):
    predicao = modelo.predict(input_scaled)
    resultado = label_encoder.inverse_transform(predicao)[0]
    st.success(f"🔎 Resultado previsto: **{resultado.replace('_', ' ')}**")
