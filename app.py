import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt  
import seaborn as sns             


# ---------- MENU LATERAL ----------
menu = st.sidebar.selectbox("📂 Navegação", [
    "📋 Avaliação Pessoal",
    "📊 Dados do Modelo",
    "📈 Análise Exploratória"
])


if menu == "📋 Avaliação Pessoal":

    # ---------- ESTILO VISUAL ----------
    st.set_page_config(page_title="Preditor de Obesidade", layout="centered")
    st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stButton>button {
            background-color: #3b5e62;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5em 1.5em;
            border: none;
        }
        .stSelectbox, .stSlider {
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---------- ESTADO DA SESSÃO ----------
    if "resultado_exibido" not in st.session_state:
        st.session_state.resultado_exibido = False
        st.session_state.resultado = None

    # ---------- CARREGAR MODELOS ----------
    modelo = joblib.load("modelo_final.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    # ---------- TÍTULO ----------
    st.title("🔍 Preditor Personalizado de Obesidade")
    st.markdown("Preencha os campos abaixo para estimar seu nível de obesidade com base nos hábitos informados:")

    # ---------- FORMULÁRIO ----------
    with st.form("formulario"):
        genero = st.selectbox("Gênero:", ["Selecione", "Feminino", "Masculino"])
        idade = st.number_input("Idade:", min_value=10, max_value=100, value=None, placeholder="Ex: 28")
        altura = st.number_input("Altura (em metros):", min_value=1.0, max_value=2.5, value=None, placeholder="Ex: 1.70")
        peso = st.number_input("Peso (em kg):", min_value=30.0, max_value=200.0, value=None, placeholder="Ex: 70")

        col1, col2 = st.columns(2)
        with col1:
            historico_familiar = st.radio("Histórico familiar de sobrepeso?", ["Sim", "Não"], horizontal=True)
            alimentos_caloricos = st.radio("Consome alimentos calóricos com frequência?", ["Sim", "Não"], horizontal=True)
            vegetais = st.slider("Frequência de vegetais na alimentação (0 = nunca, 3 = sempre)", 0.0, 3.0, 1.0)
            refeicoes_dia = st.slider("Refeições principais por dia", 1.0, 5.0, 3.0)
            lanches = st.selectbox("Costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
            agua = st.slider("Litros de água por dia", 0.0, 3.0, 2.0)
            controla_calorias = st.radio("Controla ingestão calórica?", ["Sim", "Não"], horizontal=True)
        with col2:
            atividade_fisica = st.slider("Horas de atividade física por semana", 0.0, 5.0, 1.0)
            tempo_tela = st.slider("Horas de uso de telas por dia", 0.0, 5.0, 2.0)
            transporte = st.selectbox("Meio de transporte mais usado", ["Transporte público", "A pé", "Carro", "Moto", "Bicicleta"])
            diabetes = st.radio("Já foi diagnosticado com diabetes?", ["Sim", "Não"], horizontal=True)
            pressao = st.radio("Tem pressão alta?", ["Sim", "Não"], horizontal=True)
            depressao = st.radio("Sente desânimo/falta de interesse?", ["Sim", "Não"], horizontal=True)
            alcool = st.selectbox("Frequência de consumo de álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])
            fuma = st.radio("Você fuma atualmente?", ["Sim", "Não"], horizontal=True)

        enviar = st.form_submit_button("🔍 Prever nível de obesidade")

    # ---------- PRÉ-PROCESSAMENTO ----------
    if enviar:
        if genero == "Selecione" or idade is None or altura is None or peso is None:
            st.warning("⚠️ Por favor, preencha todos os campos obrigatórios.")
        else:
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

            input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)
            input_scaled = scaler.transform(input_df)
            pred = modelo.predict(input_scaled)
            resultado = label_encoder.inverse_transform(pred)[0]
            st.session_state.resultado = resultado
            st.session_state.resultado_exibido = True

    # ---------- RESULTADO ----------
    def gerar_explicacao():
        riscos = []
        if vegetais < 1.0: riscos.append("- Baixo consumo de vegetais")
        if alimentos_caloricos == "Sim": riscos.append("- Consumo frequente de alimentos calóricos")
        if historico_familiar == "Sim": riscos.append("- Histórico familiar de obesidade")
        if atividade_fisica < 1.0: riscos.append("- Pouca atividade física")
        if alcool in ["Frequentemente", "Sempre"]: riscos.append("- Consumo elevado de álcool")
        if fuma == "Sim": riscos.append("- Tabagismo")
        if depressao == "Sim": riscos.append("- Indício de desmotivação emocional")
        if diabetes == "Sim": riscos.append("- Diabetes diagnosticado")
        if pressao == "Sim": riscos.append("- Pressão alta")
        return "Nenhum fator de risco relevante identificado." if not riscos else "\n".join(riscos)

    explicacoes_obesidade = {
        "Insufficient_Weight": "Abaixo do peso saudável. Pode indicar desnutrição ou condição metabólica.",
        "Normal_Weight": "Peso considerado saudável. Continue mantendo bons hábitos!",
        "Overweight_Level_I": "Sobrepeso leve. Requer atenção para evitar progressão à obesidade.",
        "Overweight_Level_II": "Sobrepeso moderado. Aumenta o risco de doenças metabólicas.",
        "Obesity_Type_I": "Obesidade grau I. Requer mudanças no estilo de vida.",
        "Obesity_Type_II": "Obesidade grau II. Risco elevado de doenças cardiovasculares.",
        "Obesity_Type_III": "Obesidade grau III (mórbida). Risco grave. Necessário acompanhamento médico."
    }

    def gerar_imagem_diagnostico(categoria):
        img_path = "categorias_obesidade.png"
        if not os.path.exists(img_path):
            return None
        img = Image.open(img_path).copy()
        draw = ImageDraw.Draw(img)
        categorias_coord = {
            "Insufficient_Weight": (90, 230),
            "Normal_Weight": (230, 230),
            "Overweight_Level_I": (370, 230),
            "Obesity_Type_I": (510, 230),
            "Obesity_Type_II": (650, 230),
            "Obesity_Type_III": (790, 230)
        }
        if categoria not in categorias_coord:
            return img
        x, y = categorias_coord[categoria]
        r = 80
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=6)
        return img

    if st.session_state.resultado_exibido:
        categoria = st.session_state.resultado
        descricao = explicacoes_obesidade.get(categoria, "Categoria não reconhecida.")
        st.success(f"✅ Resultado previsto: **{categoria.replace('_', ' ')}**")
        st.markdown(f"🩺 **Descrição clínica:**\n> {descricao}")
        imagem_destaque = gerar_imagem_diagnostico(categoria)
        if imagem_destaque:
            st.image(imagem_destaque, caption="🔎 Localização corporal correspondente", use_container_width=True)
        st.markdown("#### 🧠 Fatores de risco identificados:")
        st.code(gerar_explicacao())
        st.button("🔁 Fazer nova previsão", on_click=lambda: st.session_state.update({"resultado_exibido": False}))

elif menu == "📊 Dados do Modelo":
    st.title("📊 Dados do Modelo")
    st.markdown("""
### 📚 Fonte dos dados
- Base: `Obesity.csv` fornecida no desafio Tech Challenge
- Dados sobre hábitos alimentares, saúde e estilo de vida

### 🧠 Algoritmo
- `RandomForestClassifier` (modelo de classificação por árvores)
- Divisão de treino/teste: 80% treino, 20% teste
- Validação estratificada para manter a distribuição das classes

### 🎯 Acurácia geral
- **93.6%**

### 📋 Métricas por classe
| Classe               | Precisão | Recall | F1-score |
|----------------------|----------|--------|----------|
| Insufficient Weight  | 0.98     | 0.93   | 0.95     |
| Normal Weight        | 0.75     | 0.91   | 0.82     |
| Obesity Type I       | 0.97     | 0.96   | 0.96     |
| Obesity Type II      | 1.00     | 0.98   | 0.99     |
| Obesity Type III     | 1.00     | 0.98   | 0.99     |
| Overweight Level I   | 0.93     | 0.86   | 0.89     |
| Overweight Level II  | 0.96     | 0.91   | 0.94     |

### 🔗 Repositório do código
[🔗 GitHub do projeto](https://github.com/laribar/TechChallenge4)

### 🔎 Bibliotecas utilizadas
- `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `Pillow`
""")
    st.subheader("🔍 Correlação entre Variáveis")
    st.image("matriz.png", caption="Matriz de Correlação entre Variáveis Numéricas", use_container_width=True)
elif menu == "📈 Análise Exploratória":
    st.title("📈 Análise Exploratória dos Dados")

    @st.cache_data
    def carregar_base_exploratoria():
        df = pd.read_csv("obesity_personalized.csv")
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
        return df

    df = carregar_base_exploratoria()

    st.markdown("Explore os dados utilizados no modelo e visualize relações importantes entre variáveis.")

    # Distribuição da variável alvo
    st.subheader("Distribuição das Categorias de Obesidade")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Obesity", order=df["Obesity"].value_counts().index, ax=ax1)
    for p in ax1.patches:
        ax1.text(p.get_x() + p.get_width()/2, p.get_height() + 3, int(p.get_height()), ha='center')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)

    # Dispersão Idade vs Peso
    st.subheader("Idade vs Peso por Categoria de Obesidade")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="Weight", hue="Obesity", ax=ax2)
    st.pyplot(fig2)

    # Distribuição do IMC
    st.subheader("Distribuição do IMC (Índice de Massa Corporal)")
    fig3, ax3 = plt.subplots()
    sns.histplot(df["BMI"], kde=True, bins=30, ax=ax3)
    st.pyplot(fig3)

    # Matriz de Correlação
    st.subheader("Correlação entre Variáveis Numéricas")
    corr = df.select_dtypes(include='number').corr()
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)
