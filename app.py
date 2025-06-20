import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Inicializa estado da sessão
if "resultado_exibido" not in st.session_state:
    st.session_state.resultado_exibido = False
    st.session_state.resultado = None

# Carregar modelo, scaler, label encoder e colunas
modelo = joblib.load("modelo_final.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🔍 Preditor Personalizado de Obesidade")
st.write("Responda às perguntas abaixo para prever seu nível de obesidade com base em hábitos, alimentação e saúde.")

# Perguntas do questionário
genero = st.selectbox("Qual seu gênero?", ["Feminino", "Masculino"])
idade = st.slider("Qual sua idade?", 10, 100, 25)
altura = st.slider("Qual sua altura (em metros)?", 1.0, 2.5, 1.70)
peso = st.slider("Qual seu peso (em kg)?", 30.0, 200.0, 70.0)
historico_familiar = st.radio("Você tem histórico familiar de sobrepeso?", ["Sim", "Não"])
alimentos_caloricos = st.radio("Você consome alimentos calóricos com frequência?", ["Sim", "Não"])
vegetais = st.slider("Com que frequência consome vegetais nas refeições? (0 = nunca, 3 = sempre)", 0.0, 3.0, 2.0)
refeicoes_dia = st.slider("Quantas refeições principais faz por dia?", 1.0, 5.0, 3.0)
lanches = st.selectbox("Você costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
agua = st.slider("Quantos litros de água você bebe por dia?", 0.0, 3.0, 2.0)
controla_calorias = st.radio("Você controla a ingestão calórica?", ["Sim", "Não"])
atividade_fisica = st.slider("Horas de atividade física por semana:", 0.0, 5.0, 1.0)
tempo_tela = st.slider("Horas de uso de telas por dia (celular, TV, computador)", 0.0, 5.0, 2.0)
transporte = st.selectbox("Meio de transporte mais utilizado:", ["Transporte público", "A pé", "Carro", "Moto", "Bicicleta"])
diabetes = st.radio("Você já foi diagnosticado(a) com diabetes?", ["Sim", "Não"])
pressao = st.radio("Você tem pressão alta diagnosticada?", ["Sim", "Não"])
depressao = st.radio("Você tem sentido pouco interesse ou prazer nas coisas ultimamente?", ["Sim", "Não"])
alcool = st.selectbox("Com que frequência você consome bebida alcoólica?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
fuma = st.radio("Você fuma atualmente?", ["Sim", "Não"])

# Montar dicionário de entrada
input_dict = {
    "Age": idade,
    "Height": altura,
    "Weight": peso,
    "FCVC": vegetais,
    "NCP": refeicoes_dia,
    "CH2O": agua,
    "FAF": atividade_fisica,
    "TUE": tempo_tela,
    "Gender_Female": 1 if genero == "Feminino" else 0,
    "Gender_Male": 1 if genero == "Masculino" else 0,
    "family_history_yes": 1 if historico_familiar == "Sim" else 0,
    "family_history_no": 1 if historico_familiar == "Não" else 0,
    "FAVC_yes": 1 if alimentos_caloricos == "Sim" else 0,
    "FAVC_no": 1 if alimentos_caloricos == "Não" else 0,
    "CAEC_no": 1 if lanches == "Não" else 0,
    "CAEC_Sometimes": 1 if lanches == "Às vezes" else 0,
    "CAEC_Frequently": 1 if lanches == "Frequentemente" else 0,
    "CAEC_Always": 1 if lanches == "Sempre" else 0,
    "SCC_yes": 1 if controla_calorias == "Sim" else 0,
    "SCC_no": 1 if controla_calorias == "Não" else 0,
    "MTRANS_Public_Transportation": 1 if transporte == "Transporte público" else 0,
    "MTRANS_Walking": 1 if transporte == "A pé" else 0,
    "MTRANS_Automobile": 1 if transporte == "Carro" else 0,
    "MTRANS_Motorbike": 1 if transporte == "Moto" else 0,
    "MTRANS_Bike": 1 if transporte == "Bicicleta" else 0,
    "DIQ010": 1.0 if diabetes == "Sim" else 0.0,
    "MCQ160K": 1.0 if pressao == "Sim" else 0.0,
    "DPQ010": 1.0 if depressao == "Sim" else 0.0,
    "ALQ130": 0 if alcool == "Não" else 1 if alcool == "Às vezes" else 2 if alcool == "Frequentemente" else 3,
    "SMQ020": 1.0 if fuma == "Sim" else 0.0
}

# Criar DataFrame e alinhar com colunas do treino
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# Função explicativa
def gerar_explicacao():
    riscos = []
    if vegetais < 1.0:
        riscos.append("- Baixo consumo de vegetais")
    if alimentos_caloricos == "Sim":
        riscos.append("- Consumo frequente de alimentos calóricos")
    if historico_familiar == "Sim":
        riscos.append("- Histórico familiar de obesidade")
    if atividade_fisica < 1.0:
        riscos.append("- Nível de atividade física muito baixo")
    if alcool in ["Frequentemente", "Sempre"]:
        riscos.append("- Consumo elevado de bebidas alcoólicas")
    if fuma == "Sim":
        riscos.append("- Você fuma atualmente")
    if depressao == "Sim":
        riscos.append("- Indício de desmotivação/depressão")
    if diabetes == "Sim":
        riscos.append("- Diabetes diagnosticado")
    if pressao == "Sim":
        riscos.append("- Pressão alta diagnosticada")
    return "Nenhum fator de risco relevante identificado." if not riscos else "\n".join(riscos)

# Botão de previsão
if st.button("🔍 Prever nível de obesidade"):
    pred = modelo.predict(input_scaled)
    resultado = label_encoder.inverse_transform(pred)[0]
    st.session_state.resultado = resultado
    st.session_state.resultado_exibido = True

# Exibição do resultado se houver
if st.session_state.resultado_exibido:
    st.success(f"✅ Resultado previsto: **{st.session_state.resultado.replace('_', ' ')}**")
    st.markdown("#### 🧠 Fatores de risco detectados:")
    st.markdown(f"```\n{gerar_explicacao()}\n```")
    st.button("🔁 Fazer nova previsão", on_click=lambda: st.session_state.update({"resultado_exibido": False}))
