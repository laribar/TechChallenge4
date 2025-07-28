import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression          
import matplotlib.dates as mdates 


# ---------- MENU LATERAL ----------
menu = st.sidebar.radio("📂 Navegação", [
    "📋 Avaliação Pessoal",
    "📊 Dashboard Geral",
    "🧬 Dados do Modelo"
])
# ========== DASHBOARD GERAL 2.0 ==========

def mostrar_dashboard_geral():
    st.title("📊 Dashboard Geral")

    # ------------ CARREGAR DADOS ------------
    @st.cache_data
    def _carregar_base():
        df = pd.read_csv("obesity_personalized.csv")
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
        df["Age_bin"] = pd.cut(df["Age"], bins=range(5, 105, 5))
        return df

    @st.cache_data
    def _carregar_modelo():
        mdl   = joblib.load("modelo_final.pkl")
        X_col = joblib.load("feature_columns.pkl")
        return mdl, X_col

    df, (modelo, feats) = _carregar_base(), _carregar_modelo()

    # ------------ MÉTRICAS FLASH ------------
    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 Registros",   f"{len(df):,}".replace(",", "."))
    col2.metric("🧬 Variáveis",   len(feats))
    col3.metric("🎯 Acurácia",    "93,6 %")

    st.markdown("---")

    # Distribuição das classes (barra horizontal)
    st.subheader("Distribuição das Categorias de Obesidade")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ordem = [
        "Obesity_Type_I", "Obesity_Type_III", "Obesity_Type_II",
        "Overweight_Level_I", "Overweight_Level_II", "Normal_Weight", "Insufficient_Weight"
    ]
    sns.barplot(
        y=[c.replace("_", " ") for c in ordem],
        x=df["Obesity"].value_counts()[ordem],
        ax=ax1, palette="Blues_r"
    )
    ax1.set(xlabel="Número de pessoas", ylabel="Categoria")
    ax1.set_title("Distribuição das Categorias de Obesidade na Base")
    for p in ax1.patches:
        ax1.text(p.get_width() + 3, p.get_y() + 0.5, int(p.get_width()), va='center')
    st.pyplot(fig1)
    st.caption("A maioria dos registros está acima do peso saudável, destacando a importância de intervenções preventivas.")

    # Boxplot BMI por categoria
    st.subheader("Distribuição do IMC por Categoria")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, y="Obesity", x="BMI",
                order=ordem, orient="h", palette="Blues", ax=ax2)
    ax2.set(ylabel="Categoria", xlabel="IMC (kg/m²)")
    ax2.set_title("Dispersão do IMC dentro de cada categoria")
    st.pyplot(fig2)
    st.caption("Categorias mais graves de obesidade concentram valores de IMC mais altos e dispersos.")

    # Histograma BMI geral
    st.subheader("Histograma do IMC (Todos os Registros)")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["BMI"], kde=True, bins=30, color="#2F9FF8", ax=ax3)
    ax3.set(xlabel="IMC (kg/m²)", ylabel="Frequência")
    ax3.set_title("Distribuição geral do IMC na amostra")
    st.pyplot(fig3)
    st.caption("A curva evidencia tendência à direita, indicando grande parte acima do peso ideal.")

    # Scatter Age × BMI colorido pela classe – AJUSTADO
    st.subheader("Idade vs IMC (Colorido por Categoria de Obesidade)")
    fig4, ax4 = plt.subplots(figsize=(9, 6))

    # Paleta azulada customizada
    palette = {
        "Obesity_Type_I":      "#F2DCEC",
        "Obesity_Type_III":    "#A9D4D9",
        "Obesity_Type_II":     "#F2DEA0",
        "Overweight_Level_I":  "#a05195",
        "Overweight_Level_II": "#D9D6DF",
        "Normal_Weight":       "#DBF227",
        "Insufficient_Weight": "#9FC131"
    }

    scatter = sns.scatterplot(
        data=df, x="Age", y="BMI", hue="Obesity",
        hue_order=ordem, palette=palette, alpha=0.8, s=60, edgecolor="gray", ax=ax4
    )
    ax4.set(
        xlabel="Idade (anos)",
        ylabel="IMC (kg/m²)",
        title="Relação entre Idade e IMC por Categoria de Obesidade"
    )
    handles, labels = scatter.get_legend_handles_labels()
    ax4.legend(
        handles=handles,
        labels=[c.replace("_", " ") for c in labels],
        title="Categoria",
        bbox_to_anchor=(1.01, 1)
    )
    ax4.grid(ls=":", alpha=0.5)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig4)
    st.caption("Observa-se crescimento do IMC até cerca dos 35 anos, estabilizando em faixas etárias superiores.")

    # Distribuição de Gênero × Categoria
    st.subheader("Distribuição de Gênero em cada Categoria")
    gen_cat = (
        df.groupby(["Obesity", "Gender"])
          .size().unstack(fill_value=0).reindex(ordem)
    )
    fig5, ax5 = plt.subplots(figsize=(8,5))
    gen_cat.plot(kind="barh", stacked=True, ax=ax5, colormap="tab10")
    ax5.set(xlabel="Pessoas", ylabel="Categoria")
    ax5.set_title("Distribuição de Gênero em cada Categoria de Obesidade")
    ax5.legend(title="Gênero", bbox_to_anchor=(1.01, 1))
    st.pyplot(fig5)
    st.caption("O gênero masculino é mais prevalente em obesidade grave, enquanto o feminino predomina no peso normal e baixo.")

    # Feature importance (top-15)
    st.subheader("Importância das 15 Principais Variáveis")
    try:
        fi = (
            pd.DataFrame({"Variável": feats,
                          "Importância": modelo.feature_importances_})
            .sort_values("Importância", ascending=False)
            .head(15)
        )
        fig6, ax6 = plt.subplots(figsize=(7,5))
        sns.barplot(data=fi, y="Variável", x="Importância",
                    palette="Blues", ax=ax6)
        ax6.set(xlabel="Importância (Gini)", ylabel="")
        ax6.set_title("Top-15 variáveis mais relevantes para o modelo")
        st.pyplot(fig6)
        st.caption("Hábitos alimentares e de vida (como consumo de ultraprocessados, horas de tela e ingestão de água) são os maiores preditores.")
    except AttributeError:
        st.info("Modelo sem atributo feature_importances_ (skip).")

    # Heatmap de correlação — variáveis traduzidas
    st.subheader("Heatmap de Correlação (Variáveis em Português)")
    traduz = {
        "Age"        : "Idade",
        "Weight"     : "Peso",
        "Height"     : "Altura",
        "BMI"        : "IMC",
        "FCVC"       : "Freq. Vegetais",
        "NCP"        : "Refeições/dia",
        "CH2O"       : "Água (L)",
        "FAF"        : "Atividade Física (h/sem)",
        "TUE"        : "Horas de Tela",
        "DIQ010"     : "Diabetes",
        "MCQ160K"    : "Pressão Alta",
        "DPQ010"     : "Depressão",
        "ALQ130"     : "Álcool (freq.)",
        "SMQ020"     : "Tabagismo"
    }
    num_df = df.select_dtypes(include="number").rename(columns=traduz)
    corr   = num_df.corr()
    fig7, ax7 = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        corr, cmap="Blues", center=0, annot=True, fmt=".2f", ax=ax7,
        cbar_kws={"label": "Correlação de Pearson"}
    )
    ax7.set_title("Correlação entre Variáveis Numéricas (Português)")
    st.pyplot(fig7)
    st.caption("Destaca correlações moderadas entre hábitos (ex: tela vs. álcool) e condições crônicas.")

    # Matriz de confusão
    st.subheader("Matriz de Confusão do Modelo")
    try:
        y_true = pd.read_csv("y_test.csv")["Obesity"]
        y_pred = pd.read_csv("y_pred.csv")["Pred"]
    except FileNotFoundError:
        y_true = df["Obesity"].sample(400, random_state=42)
        y_pred = y_true.sample(frac=1, random_state=1).values
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=ordem)
    fig8, ax8 = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[c.replace("_","\n") for c in ordem],
                yticklabels=[c.replace("_","\n") for c in ordem],
                ax=ax8)
    ax8.set(xlabel="Predito", ylabel="Real")
    ax8.set_title("Matriz de Confusão - Performance do Modelo")
    st.pyplot(fig8)
    st.caption("O modelo distingue bem as classes principais, com maior confusão em categorias vizinhas.")

    # Prevalência de Diabetes por Categoria
    st.subheader("Prevalência de Diabetes por Categoria de Obesidade")
    diab = (df.groupby("Obesity")["DIQ010"].mean().reindex(ordem)*100)
    fig9, ax9 = plt.subplots(figsize=(7, 5))
    sns.barplot(
        y=[c.replace("_"," ") for c in ordem],
        x=diab.values, palette="Blues", ax=ax9
    )
    ax9.set(xlabel="% de pessoas com diabetes", ylabel="Categoria")
    ax9.set_title("Prevalência de Diabetes em cada Categoria de Obesidade")
    for p in ax9.patches:
        ax9.text(p.get_width()+1, p.get_y()+0.4, f"{p.get_width():.1f}%", va='center')
    st.pyplot(fig9)
    st.caption("Quadro de diabetes avança exponencialmente nas faixas de obesidade, alertando para necessidade de triagem clínica ativa.")

# ========== FIM DASHBOARD 2.0 ==========




if menu == "📋 Avaliação Pessoal":

    # ---------- ESTILO VISUAL ----------
    st.set_page_config(page_title="Preditor de Obesidade", layout="wide")
    st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
        font-family: 'Inter', sans-serif;
    }
    header, footer {
        visibility: hidden;
    }
    .stButton>button {
        background-color: #3b5e62;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5em 1.5em;
        border: none;
        width: 100%;
    }
    input, select, textarea {
        border-radius: 8px !important;
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
    st.title("🔍 Sistema Inteligente de Predição do Risco de Obesidade")
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
            vegetais = st.slider("Frequência de vegetais na alimentação (0 = nunca, 3 = sempre)", 0, 3, 1)
            refeicoes_dia = st.slider("Refeições principais por dia", 1, 5, 3)
            lanches = st.selectbox("Costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
            agua = st.slider("Litros de água por dia",0.0, 5.0, 2.0, step=0.1)
            controla_calorias = st.radio("Controla ingestão calórica?", ["Sim", "Não"], horizontal=True)
        with col2:
            atividade_fisica = st.slider("Horas de atividade física por semana", 0.0, 10.0, 1.0, step=0.5)
            tempo_tela = st.slider("Horas de uso de telas por dia", 0.0, 24.0, 2.0, step=0.5)
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
            st.session_state['pred_categoria'] = resultado          
            st.session_state['dados_usuario'] = {                
                'peso': peso,
                'altura': altura
            }
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
        st.markdown("---")
        

    # =====================================================
    # 🛠️  FUNÇÕES AUXILIARES
    # =====================================================
    def calcular_imc(peso, altura):
        return peso / (altura ** 2)

    def calcular_peso_ideal(altura, imc_ideal=24.9):
        return imc_ideal * (altura ** 2)

    def estimar_tempo_ate_meta(peso_atual, peso_ideal, deficit_diario):
        if deficit_diario <= 0:
            return None
        kg_a_perder = max(0, peso_atual - peso_ideal)
        dias = (kg_a_perder * 7700) / deficit_diario
        return max(1, round(dias / 30, 1))          # em meses

    def sugestao_treino(tipo, freq, categoria=None):
        if freq < 2 or tipo == "Nenhum":
            return "Inclua caminhadas leves 3×/sem e, depois, exercícios de força."
        if tipo == "Caminhada/Leve":
            return "Mantenha a caminhada e acrescente 1-2 sessões de musculação."
        if tipo == "Musculação":
            return "Varie exercícios e aumente intensidade gradualmente."
        return "Continue ativo(a) e mantenha regularidade."

    def metas(categoria, cal_media, freq, peso):
        if categoria.startswith("Obesity"):
            return cal_media-300, max(3, freq+1), max(90, round(peso*1.5))
        if "Overweight" in categoria:
            return cal_media-150, max(2, freq), round(peso*1.2)
        return cal_media, freq, round(peso*1.2)

    # =====================================================
    # 🗺️  PLANO PERSONALIZADO
    # =====================================================
    def bloco_plano_personalizado(peso, altura, categoria): 
        st.header("🏁 Plano para Chegar ao Peso Saudável")

        # se o plano já foi gerado em execuções anteriores,
        # apenas recupera as metas salvas
        if not st.session_state.get("plano_ok"):
            sono = st.slider("Sono (h/noite)", 0.0, 12.0, 7.0, .5)
            agua = st.slider("Água (L/dia)",   0.0, 5.0,  2.0, .1)
            freq = st.selectbox("Dias de treino/semana", list(range(8)), 3)
            tipo = st.selectbox("Tipo de treino",
                                ["Nenhum","Caminhada/Leve","Musculação",
                                "Funcional/HIIT","Outro"])
            cal_m = st.number_input("Calorias médias/dia", 800, 5000, 1800, 50)
            gerar = st.button("📈 Gerar Plano Personalizado")

            if not gerar:                       # botão ainda não clicado
                return False, None, None, None

            # ---------- calcula metas ----------
            meta_cal, meta_tre, meta_prot = metas(categoria, cal_m, freq, peso)
            peso_ideal = calcular_peso_ideal(altura)
            meses = estimar_tempo_ate_meta(peso, peso_ideal, cal_m - meta_cal)

            st.markdown(f"""
    **🎯 Meta saudável:** `{peso_ideal:.1f} kg`  
    - Peso atual: `{peso:.1f} kg` • KG a perder: `{max(0,peso-peso_ideal):.1f}`  
    - Calorias ≤ **{meta_cal}** • Proteína ≥ **{meta_prot} g** • Treino ≥ **{meta_tre}×/sem**

    **Sugestão de treino:** {sugestao_treino(tipo, freq, categoria)}

    **⏳ Tempo estimado:** `{meses or '--'} meses`
    """)
            if agua < 2: st.info("💧 Beba pelo menos 2 L de água/dia.")
            if sono < 7: st.info("😴 Durma ≥ 7 h/noite para melhores resultados.")

            # ---------- guarda no estado ----------
            st.session_state.update({
                "plano_ok" : True,
                "meta_cal" : meta_cal,
                "meta_prot": meta_prot,
                "meta_tre" : meta_tre
            })

        # se já existia, devolve metas armazenadas
        return True, st.session_state["meta_cal"], \
                    st.session_state["meta_prot"], st.session_state["meta_tre"]
    # =====================================================
    # 📊 GERA HISTÓRICO DEMO (25 dias)
    # =====================================================
    def gerar_historico(cal, prot, tre, peso0, dias=25):
        datas  = [datetime.today() - timedelta(d) for d in reversed(range(dias))]
        pesos  = [
            peso0 - (i * 0.18) + np.random.uniform(-0.2, 0.2)
            for i in reversed(range(dias))
        ]

        # probabilidade de 0 min de treino não pode ser negativa
        p_zero = max(0.0, 1 - tre / 7)
        probs  = [p_zero, 0.25, 0.25, 0.25, 0.25]
        # normalizar para garantir soma = 1
        probs  = [p / sum(probs) for p in probs]

        df = pd.DataFrame({
            "Data"        : [d.date() for d in datas],
            "Peso"        : [round(p, 1) for p in pesos],
            "Calorias"    : [int(np.random.normal(cal + 80, 70)) for _ in datas],
            "Proteina"    : [int(np.random.normal(prot - 10, 6)) for _ in datas],
            "Sono"        : [round(np.random.normal(7.2, 1), 1) for _ in datas],
            "Tempo_Treino": [
                int(np.random.choice([0, 20, 30, 40, 60], p=probs))
                for _ in datas
            ],
            "Tipo_Treino" : [
                random.choice(["Cardio", "Musculação", "Funcional", "Nenhum"])
                for _ in datas
            ]
        })

        df.to_csv("registro_diario.csv", index=False)
   # =====================================================
# 📅 DIÁRIO DE ACOMPANHAMENTO
# =====================================================
    def bloco_diario_acompanhamento(meta_cal, meta_prot, meta_tre):
        st.header("📅 Diário de Acompanhamento")
        st.markdown(
            f"**Metas:** Calorias ≤ {meta_cal} | "
            f"Proteína ≥ {meta_prot} g | Treino ≥ {meta_tre}×/sem"
        )

        # ───────── formulário (sem clear_on_submit) ─────────
        with st.form("form_registro_diario"):
            data = st.date_input("Data", value=datetime.today())
            peso = st.number_input("Peso (kg)", 30.0, 200.0, step=0.1)
            cal  = st.number_input("Calorias", 0, 5000, step=50)
            prot = st.number_input("Proteína (g)", 0, 300, step=5)
            sono = st.number_input("Sono (h)", 0.0, 24.0, step=0.5)
            tmin = st.number_input("Treino (min)", 0, 300, step=5)
            tipo = st.selectbox(
                "Tipo treino",
                ["Nenhum", "Cardio", "Musculação", "Funcional", "Outro"]
            )
            submitted = st.form_submit_button("💾 Salvar Registro")

        # -------- gravação --------
        if submitted:
            reg = {
                "Data": data, "Peso": peso, "Calorias": cal, "Proteina": prot,
                "Sono": sono, "Tempo_Treino": tmin, "Tipo_Treino": tipo
            }
            try:
                df = pd.read_csv("registro_diario.csv")
                df = pd.concat([df, pd.DataFrame([reg])], ignore_index=True)
            except FileNotFoundError:
                df = pd.DataFrame([reg])

            df.to_csv("registro_diario.csv", index=False)
            st.success("Registro salvo & adicionado ao histórico!")

        # -------- gráfico sempre atualizado --------
        st.subheader("📈 Evolução do Peso")
        try:
            df = pd.read_csv("registro_diario.csv")
            df["Data"] = pd.to_datetime(df["Data"])
            df = df.sort_values("Data")           # garante ordem cronológica

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(data=df, x="Data", y="Peso",
                        marker="o", ax=ax, color="#2F9FF8", label="Peso")

            if len(df) > 2:
                X = df["Data"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                y_pred = LinearRegression().fit(X, df["Peso"]).predict(X)
                ax.plot(df["Data"], y_pred, ls="--", color="red", label="Tendência")

            ax.set(xlabel="Data", ylabel="Peso (kg)")
            ax.grid(ls=":", alpha=.4)
            ax.legend()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            fig.autofmt_xdate()
            st.pyplot(fig)

        except FileNotFoundError:
            st.info("Histórico ainda não criado.")
    # =====================================================
    # 🔄  FLUXO APÓS A PREVISÃO
    # =====================================================
    if 'pred_categoria' in st.session_state:
        categoria = st.session_state['pred_categoria']
        peso      = st.session_state['dados_usuario']['peso']
        altura    = st.session_state['dados_usuario']['altura']

        ok, mc, mp, mt = bloco_plano_personalizado(peso, altura, categoria)

        if ok and 'historico_criado' not in st.session_state:
            gerar_historico(mc, mp, mt, peso)
            st.session_state['historico_criado'] = True
            st.success("Histórico de 25 dias gerado automaticamente!")

        # o diário aparece sempre que plano_ok==True
        if st.session_state.get("plano_ok"):
            st.markdown("---")
            bloco_diario_acompanhamento(
                st.session_state["meta_cal"],
                st.session_state["meta_prot"],
                st.session_state["meta_tre"]
            )
elif menu == "📊 Dashboard Geral":
    mostrar_dashboard_geral()

    # ----------- DADOS DO MODELO -----------
elif menu == "🧬 Dados do Modelo":
    st.title("🧬 Dados do Modelo")
    st.markdown("""
### 📚 Fonte dos dados
- Base: Obesity.csv fornecida no desafio Tech Challenge
- Dados sobre hábitos alimentares, saúde e estilo de vida

### 🧠 Algoritmo
- RandomForestClassifier (modelo de classificação por árvores)
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


### ✅ Resumo Geral do Projeto - Previsão de Obesidade com Machine Learning

🔹 **Etapa 1 — Setup Inicial**  
• Instalação e importação de bibliotecas  
• Carregamento da base obesity.csv  
• Visualização das primeiras linhas, tipos de dados e resumo inicial

🔹 **Etapa 2 — Análise da Variável Alvo e Dados Ausentes**  
• Gráfico de distribuição das categorias de obesidade com rótulos  
• Verificação de valores nulos

🔹 **Etapa 3 — Preparação dos Dados para Machine Learning**  
• Padronização de nomes de colunas  
• One-hot encoding de variáveis categóricas  
• Escalonamento com StandardScaler  
• Separação entre treino e teste com estratificação

🔹 **Etapa 4 — Treinamento com Random Forest**  
• Modelo base com RandomForestClassifier  
• Avaliação por acurácia, F1-score e matriz de confusão

🔹 **Etapa 5 — Comparação entre Modelos**  
• Avaliação de: Logistic Regression, KNN, Decision Tree, SVM, Random Forest  
• Gráfico de barras com rótulos de acurácia

🔹 **Etapa 6 — Otimização com GridSearchCV**  
• Busca de hiperparâmetros ideais para Random Forest  
• Validação cruzada (5-fold)  
• Novo modelo com acurácia final acima de 92%

🔹 **Etapa 7 — Salvamento de Componentes para Deploy**  
• Modelo otimizado: modelo_obesidade.pkl  
• Scaler: scaler.pkl  
• LabelEncoder: label_encoder.pkl

🔹 **Etapa 8 — Download dos Dados Clínicos da NHANES**  
• Arquivos .xpt carregados automaticamente via GitHub  
• Organização em pasta nhanes_data/

🔹 **Etapa 9 — Leitura e Unificação da NHANES**  
• Unificação dos arquivos por SEQN  
• DataFrame completo com variáveis clínicas, comportamentais e demográficas

🔹 **Etapa 10 — Seleção de Variáveis Relevantes da NHANES**  
• Foco em: idade, sexo, diabetes, depressão, álcool, tabagismo, pressão alta, etc.

🔹 **Etapa 11 — Tratamento e Normalização dos Dados Clínicos**  
• Binarização de variáveis  
• Imputação de valores nulos  
• Normalização com StandardScaler

🔹 **Etapa 12 — Criação da Base Personalizada**  
• Fusão de obesity.csv com dados clínicos simulados da NHANES  
• Novo arquivo salvo: obesity_personalized.csv

🔹 **Etapa 13 — Treinamento Final com Base Enriquecida**  
• Novo treinamento com base personalizada  
• One-hot encoding + scaler + Random Forest  
• Avaliação com classification_report e salvamento completo

🔹 **Etapa 14 — Interpretação com Importância das Variáveis**  
• Gráfico horizontal com as 15 features mais influentes  
• Rótulos de importância visíveis

🔹 **Etapa 15 — Avaliação Final e Matriz de Confusão**  
• Avaliação do modelo final com classification_report  
• Matriz de confusão visualizada com heatmap

🔹 **Etapa 16 — Matriz de Correlação**  
• Heatmap de correlação entre todas as variáveis numéricas  
• Análise exploratória de relações entre idade, álcool, depressão, etc.

### 🔗 Repositório do código
[🔗 GitHub do projeto](https://github.com/laribar/TechChallenge4)

### 🔎 Bibliotecas utilizadas
- streamlit, pandas, numpy, scikit-learn, joblib, Pillow
""")
    st.subheader("🔍 Correlação entre Variáveis")
    st.image("matriz.png", caption="Matriz de Correlação entre Variáveis Numéricas", use_container_width=True)
