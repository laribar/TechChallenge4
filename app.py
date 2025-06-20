# Este é o app Streamlit que irá carregar o modelo salvo, coletar os dados do usuário,
# aplicar as mesmas transformações do pipeline (normalização e codificação)
# e retornar a predição do nível de obesidade com uma interface simples e intuitiva.

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar modelo, scaler e label encoder
model = joblib.load("modelo_obesidade.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🔍 Preditor de Obesidade")
st.write("Preencha os dados abaixo para prever o nível de obesidade.")

# Inputs do usuário
gender = st.selectbox("Gênero", ["Male", "Female"])
age = st.slider("Idade", 10, 100, 25)
height = st.slider("Altura (em metros)", 1.0, 2.5, 1.70)
weight = st.slider("Peso (em kg)", 30.0, 200.0, 70.0)
family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"])
favc = st.selectbox("Consome alimentos altamente calóricos com frequência?", ["yes", "no"])
fcvc = st.slider("Consome vegetais nas refeições (0 = nunca, 3 = sempre)", 0.0, 3.0, 2.0)
ncp = st.slider("Número de refeições por dia", 1.0, 5.0, 3.0)
caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Fuma?", ["yes", "no"])
ch2o = st.slider("Consumo de água diário (litros)", 0.0, 3.0, 2.0)
scc = st.selectbox("Controla calorias ingeridas?", ["yes", "no"])
faf = st.slider("Atividade física semanal (horas)", 0.0, 5.0, 1.0)
tue = st.slider("Tempo de uso de tecnologia (horas por dia)", 0.0, 5.0, 2.0)
calc = st.selectbox("Frequência de consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Meio de transporte usado com mais frequência", 
                      ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Organizar dados em DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "FCVC": [fcvc],
    "NCP": [ncp],
    "CH2O": [ch2o],
    "FAF": [faf],
    "TUE": [tue],
    "Gender_Female": [1 if gender == "Female" else 0],
    "Gender_Male": [1 if gender == "Male" else 0],
    "family_history_no": [1 if family_history == "no" else 0],
    "family_history_yes": [1 if family_history == "yes" else 0],
    "FAVC_no": [1 if favc == "no" else 0],
    "FAVC_yes": [1 if favc == "yes" else 0],
    "CAEC_Always": [1 if caec == "Always" else 0],
    "CAEC_Frequently": [1 if caec == "Frequently" else 0],
    "CAEC_Sometimes": [1 if caec == "Sometimes" else 0],
    "CAEC_no": [1 if caec == "no" else 0],
    "SMOKE_no": [1 if smoke == "no" else 0],
    "SMOKE_yes": [1 if smoke == "yes" else 0],
    "SCC_no": [1 if scc == "no" else 0],
    "SCC_yes": [1 if scc == "yes" else 0],
    "CALC_Always": [1 if calc == "Always" else 0],
    "CALC_Frequently": [1 if calc == "Frequently" else 0],
    "CALC_Sometimes": [1 if calc == "Sometimes" else 0],
    "CALC_no": [1 if calc == "no" else 0],
    "MTRANS_Automobile": [1 if mtrans == "Automobile" else 0],
    "MTRANS_Bike": [1 if mtrans == "Bike" else 0],
    "MTRANS_Motorbike": [1 if mtrans == "Motorbike" else 0],
    "MTRANS_Public_Transportation": [1 if mtrans == "Public_Transportation" else 0],
    "MTRANS_Walking": [1 if mtrans == "Walking" else 0]
})

# Normalizar os dados de entrada
input_scaled = scaler.transform(input_data)

# Fazer a predição
if st.button("Prever Obesidade"):
    prediction = model.predict(input_scaled)
    resultado = label_encoder.inverse_transform(prediction)[0]
    st.success(f"💡 Resultado previsto: **{resultado.replace('_', ' ')}**")
