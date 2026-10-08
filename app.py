import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
CAMINHO_MODELO = BASE / "modelo" / "modelo.pkl"
CAMINHO_META = BASE / "modelo" / "metadados.json"
CAMINHO_EXEMPLOS = BASE / "modelo" / "exemplos_consistencia.csv"

OPCAO_MANUAL = "Entrada manual (medianas do treino)"
TOLERANCIA_PARIDADE = 1e-4

st.set_page_config(page_title="Detecção de Fraude em Cartão de Crédito", layout="wide")


@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


@st.cache_data
def carregar_metadados():
    with open(CAMINHO_META, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def carregar_exemplos():
    return pd.read_csv(CAMINHO_EXEMPLOS)


arquivos_ausentes = [p.name for p in (CAMINHO_MODELO, CAMINHO_META, CAMINHO_EXEMPLOS) if not p.exists()]
if arquivos_ausentes:
    st.error(
        f"Arquivos não encontrados em modelo/: {', '.join(arquivos_ausentes)}. "
        "Execute o notebook completo para gerar os artefatos antes de abrir o aplicativo."
    )
    st.stop()

pipeline = carregar_modelo()
meta = carregar_metadados()
exemplos = carregar_exemplos()

features = meta["features"]
ranges = meta["ranges_treino"]
medianas = meta["medianas_treino"]
limiar = float(meta["limiar_decisao"])


def chave(col):
    return f"in_{col}"


def converter(col, valor):
    return int(round(valor)) if col == "hora_relativa" else float(valor)


def restaurar_medianas():
    for col in features:
        st.session_state[chave(col)] = converter(col, medianas[col])


def aplicar_selecao():
    escolha = st.session_state["seletor_exemplo"]
    if escolha == OPCAO_MANUAL:
        restaurar_medianas()
        return
    linha = exemplos[exemplos["rotulo"] == escolha].iloc[0]
    for col in features:
        st.session_state[chave(col)] = converter(col, linha[col])


for col in features:
    st.session_state.setdefault(chave(col), converter(col, medianas[col]))
st.session_state.setdefault("seletor_exemplo", OPCAO_MANUAL)

with st.sidebar:
    st.header("Sobre o modelo")
    st.write(f"**Configuração:** {meta['modelo_nome']}")
    st.write(f"**Métrica principal:** {meta['metrica_principal']}")
    st.write(f"**Treinado em:** {meta['data_treino']}")
    st.write(f"**Limiar de decisão:** {limiar:.4f}")

    mt = meta["metricas_teste"]
    st.subheader("Métricas no conjunto de teste")
    st.metric("PR-AUC", f"{mt['PR-AUC (AP)']:.4f}")
    st.metric("ROC-AUC", f"{mt['ROC-AUC']:.4f}")
    c1, c2 = st.columns(2)
    c1.metric("Precision", f"{mt['Precision']:.3f}")
    c2.metric("Recall", f"{mt['Recall']:.3f}")
    st.metric("F1", f"{mt['F1']:.3f}")

    mc = meta["metricas_cv"]
    st.caption(f"Validação cruzada (treino): PR-AUC = {mc['AP_media']:.4f} +/- {mc['AP_dp']:.4f}")
    st.caption(f"Prevalência de fraude no treino: {meta['prevalencia_treino'] * 100:.3f}%")

st.title("Detecção de Fraude em Transações de Cartão de Crédito")
st.caption(
    "Checkpoint 5 — Data Science & Statistical Computing (FIAP, 2026). "
    "O aplicativo usa o mesmo pipeline salvo pelo notebook, sem refazer o pré-processamento."
)

st.subheader("1. Dados da transação")

st.selectbox(
    "Origem dos dados",
    [OPCAO_MANUAL] + list(exemplos["rotulo"]),
    key="seletor_exemplo",
    on_change=aplicar_selecao,
    help="Escolha um caso do teste de consistência do notebook para reproduzir a previsão e verificar a paridade.",
)

col_a, col_b = st.columns(2)
with col_a:
    st.slider("Hora relativa ao início da captura (0 a 23)", min_value=0, max_value=23, key=chave("hora_relativa"))
with col_b:
    st.number_input("Valor da transação (Amount)", min_value=0.0, step=1.0, format="%.4f", key=chave("Amount"))

with st.expander("Componentes anonimizados (V1 a V28)", expanded=False):
    st.caption("Componentes principais fornecidos pelo provedor da base. Não possuem interpretação direta.")
    colunas_v = [c for c in features if c.startswith("V")]
    grade = st.columns(4)
    for i, col in enumerate(colunas_v):
        with grade[i % 4]:
            st.number_input(col, step=0.1, format="%.8f", key=chave(col))

st.button("Restaurar medianas do treino", on_click=restaurar_medianas)

entrada = pd.DataFrame(
    [{col: st.session_state[chave(col)] for col in features}],
    columns=features,
)

fora_do_range = []
for col in features:
    valor = float(entrada.loc[0, col])
    if valor < ranges[col]["min"] or valor > ranges[col]["max"]:
        fora_do_range.append(
            f"{col} = {valor:.4f} (treino: {ranges[col]['min']:.4f} a {ranges[col]['max']:.4f})"
        )

st.subheader("2. Previsão")

if fora_do_range:
    st.warning(
        "Alerta de extrapolação: os valores abaixo estão fora do intervalo observado no treino, "
        "e a previsão pode ser menos confiável.\n\n- " + "\n- ".join(fora_do_range)
    )

proba = float(pipeline.predict_proba(entrada)[0, 1])
eh_fraude = proba >= limiar

col1, col2, col3 = st.columns(3)
col1.metric("Probabilidade de fraude", f"{proba:.2%}")
col2.metric("Limiar de decisão", f"{limiar:.2%}")
col3.metric("Razão em relação ao limiar", f"{proba / limiar:.2f}x")

st.progress(min(max(proba, 0.0), 1.0))

if eh_fraude:
    st.error("Transação classificada como SUSPEITA DE FRAUDE: encaminhar para revisão ou bloqueio.")
else:
    st.success("Transação classificada como LEGÍTIMA.")

st.caption(
    "A probabilidade de fraude é a saída de predict_proba. A decisão usa o limiar definido no treino "
    "(máximo F1 sobre previsões out-of-fold), e não o valor fixo 0,5."
)

escolha_atual = st.session_state["seletor_exemplo"]
if escolha_atual != OPCAO_MANUAL:
    st.subheader("3. Paridade com o notebook")
    linha = exemplos[exemplos["rotulo"] == escolha_atual].iloc[0]
    entradas_iguais = all(
        np.isclose(float(entrada.loc[0, col]), float(linha[col]), rtol=0.0, atol=1e-8) for col in features
    )
    if entradas_iguais:
        proba_nb = float(linha["proba_notebook"])
        diferenca = abs(proba - proba_nb)
        p1, p2, p3 = st.columns(3)
        p1.metric("Probabilidade no notebook", f"{proba_nb:.6f}")
        p2.metric("Probabilidade no aplicativo", f"{proba:.6f}")
        p3.metric("Diferença absoluta", f"{diferenca:.2e}")
        st.write(
            f"Valor real no teste (y): **{int(linha['y_real'])}** | "
            f"previsão no notebook: **{int(linha['previsao_notebook'])}**"
        )
        if diferenca < TOLERANCIA_PARIDADE:
            st.success("Paridade confirmada: a previsão do aplicativo coincide com a do notebook.")
        else:
            st.error("Divergência entre o aplicativo e o notebook. Verifique as versões das bibliotecas.")
    else:
        st.info("Os valores foram alterados em relação ao caso escolhido, então a comparação com o notebook não se aplica.")

with st.expander("Valores enviados ao modelo"):
    st.dataframe(entrada.T.rename(columns={0: "valor"}))