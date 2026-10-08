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


MOEDA = "€"

st.set_page_config(page_title="Detector de Fraude", page_icon="🛡️", layout="centered")


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


ausentes = [p.name for p in (CAMINHO_MODELO, CAMINHO_META, CAMINHO_EXEMPLOS) if not p.exists()]
if ausentes:
    st.error(f"Arquivos não encontrados em modelo/: {', '.join(ausentes)}. Execute o notebook completo antes.")
    st.stop()

pipeline = carregar_modelo()
meta = carregar_metadados()
exemplos = carregar_exemplos()
features = meta["features"]
limiar = float(meta["limiar_decisao"])
mt = meta["metricas_teste"]


def dinheiro(v):
    return f"{MOEDA} {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def prever(linha_base, valor, hora):
    x = pd.DataFrame([linha_base[features].astype(float)], columns=features)
    x["Amount"] = float(valor)
    x["hora_relativa"] = float(hora)
    return float(pipeline.predict_proba(x)[0, 1])



with st.sidebar:
    st.header("Como o modelo se saiu")
    st.caption("Medido em transações que ele nunca tinha visto (conjunto de teste).")
    st.metric("Alertas que eram fraude de verdade", f"{mt['Precision']:.0%}",
              help="Precision: de cada 100 alertas, quantos eram fraude.")
    st.metric("Fraudes que o modelo pegou", f"{mt['Recall']:.0%}",
              help="Recall: de cada 100 fraudes reais, quantas foram apanhadas.")
    with st.expander("Detalhes técnicos"):
        st.write(f"**Modelo:** {meta['modelo_nome']}")
        st.write(f"**PR-AUC:** {mt['PR-AUC (AP)']:.4f}")
        st.write(f"**ROC-AUC:** {mt['ROC-AUC']:.4f}")
        st.write(f"**F1:** {mt['F1']:.3f}")
        st.write(f"**Limiar de decisão:** {limiar:.4f}")
        st.write(f"**Fraude na base de treino:** {meta['prevalencia_treino'] * 100:.3f}%")
        st.write(f"**Treinado em:** {meta['data_treino']}")


st.title("Esta transação é fraude?")
st.write(
    "O modelo analisa uma compra de cartão e diz a **chance de ser fraude**. "
    "Se passar de **{:.0%}**, o sistema levanta um alerta.".format(limiar)
)


st.subheader("1. Escolha uma transação")
letras = "ABCDEFGHIJ"
ids = list(range(len(exemplos)))


def nome_caso(i):
    r = exemplos.iloc[i]
    return f"{letras[i]} · {dinheiro(r['Amount'])} às {int(r['hora_relativa']):02d}h"


caso = st.radio("Transação", ids, format_func=nome_caso, horizontal=True, label_visibility="collapsed")
linha = exemplos.iloc[caso]
k_valor, k_hora = f"valor_{caso}", f"hora_{caso}"
st.session_state.setdefault(k_valor, float(linha["Amount"]))
st.session_state.setdefault(k_hora, int(linha["hora_relativa"]))


def voltar_original():
    st.session_state[k_valor] = float(linha["Amount"])
    st.session_state[k_hora] = int(linha["hora_relativa"])



st.subheader("2. Valor e horário")
c1, c2 = st.columns(2)
valor = c1.number_input(f"Valor da compra ({MOEDA})", min_value=0.0, max_value=25000.0, step=10.0,
                        format="%.2f", key=k_valor)
hora = c2.slider("Horário (hora do dia, 0 a 23)", 0, 23, key=k_hora)
st.button("↺ Voltar aos valores originais", on_click=voltar_original)


proba = prever(linha, valor, hora)
eh_fraude = proba >= limiar

st.subheader("3. Resultado")
r1, r2 = st.columns([1, 1])
r1.metric("Chance de ser fraude", f"{proba:.1%}")
with r2:
    if eh_fraude:
        st.error("### 🚨 Possível fraude\nEncaminhar para revisão ou bloqueio.")
    else:
        st.success("### ✅ Parece legítima\nPode seguir normalmente.")
st.progress(min(max(proba, 0.0), 1.0))
st.caption(f"Alerta dispara a partir de {limiar:.0%}.")


if st.toggle("Mostrar a resposta real"):
    real_fraude = int(linha["y_real"]) == 1
    acertou = real_fraude == eh_fraude
    st.write(f"Na vida real, esta transação **{'era fraude' if real_fraude else 'era legítima'}**.")
    if acertou:
        st.success("O modelo acertou.")
    else:
        st.warning("O modelo errou neste caso. Nenhum modelo acerta tudo.")
    st.caption(f"Tipo de caso: {linha['rotulo']}")


with st.expander("Por que o valor e o horário quase não mudam o resultado?"):
    st.write(
        "A decisão do modelo vem principalmente de **28 características anonimizadas** (V1 a V28) "
        "que o banco calcula a partir dos dados do cartão e da compra. Elas ficam fixas em cada "
        "transação de exemplo, e por isso você só mexe em valor e horário. "
        "Esses dois campos ajudam pouco. Repare que mudar o valor para qualquer número "
        "raramente inverte o veredito."
    )
    montantes = np.geomspace(0.5, 5000, 30)
    curva = pd.DataFrame({"Chance de fraude": [prever(linha, m, hora) for m in montantes]},
                         index=pd.Index(np.round(montantes, 2), name=f"Valor ({MOEDA})"))
    st.line_chart(curva, y_label="Chance de fraude", height=220)
    st.caption("Chance de fraude desta transação ao variar só o valor (horário mantido).")

with st.expander("Como funciona, em uma frase"):
    st.write(
        "Um Random Forest (muitas árvores de decisão votando) aprendeu, com centenas de milhares de transações "
        "reais, quais combinações de características aparecem em fraudes, e devolve a fração de "
        "árvores que votam 'fraude'. Só cerca de 0,17% das transações da base são fraude."
    )