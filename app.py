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
TOLERANCIA_PARIDADE = 1e-4
TIPICO = "tipico"

st.set_page_config(page_title="Detector de Fraude", page_icon="🛡️", layout="centered")


@st.cache_resource
def carregar_modelo(versao):
    return joblib.load(CAMINHO_MODELO)


@st.cache_data
def carregar_metadados(versao):
    with open(CAMINHO_META, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def carregar_exemplos(versao):
    return pd.read_csv(CAMINHO_EXEMPLOS)


def versao_arquivo(caminho):
    return caminho.stat().st_mtime_ns if caminho.exists() else 0


ausentes = [p.name for p in (CAMINHO_MODELO, CAMINHO_META, CAMINHO_EXEMPLOS) if not p.exists()]
if ausentes:
    st.error(f"Arquivos não encontrados em modelo/: {', '.join(ausentes)}. Execute o notebook completo antes.")
    st.stop()

pipeline = carregar_modelo(versao_arquivo(CAMINHO_MODELO))
meta = carregar_metadados(versao_arquivo(CAMINHO_META))
exemplos = carregar_exemplos(versao_arquivo(CAMINHO_EXEMPLOS))
features = meta["features"]
limiar = float(meta["limiar_decisao"])
mt = meta["metricas_teste"]
medianas = meta["medianas_treino"]
valor_max = float(meta["ranges_treino"]["Amount"]["max"])
algoritmo = meta.get("algoritmo", "")


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
    f"O modelo analisa uma compra de cartão e dá uma **pontuação de risco** de fraude. "
    f"Se passar de **{limiar:.0%}**, o sistema levanta um alerta."
)

st.subheader("1. Escolha o perfil da transação")
st.caption(
    "O perfil reúne as 28 características anonimizadas do cartão e da compra (V1 a V28), "
    "que são o que mais pesa na decisão. Valor e horário você ajusta no passo 2."
)
letras = "ABCDEFGHIJ"
opcoes = list(range(len(exemplos))) + [TIPICO]


def nome_caso(i):
    if i == TIPICO:
        return "Perfil típico (compra comum)"
    r = exemplos.iloc[i]
    return f"{letras[i]} · {dinheiro(r['Amount'])} às {int(r['hora_relativa']):02d}h"


caso = st.radio("Perfil", opcoes, format_func=nome_caso, horizontal=True, label_visibility="collapsed")

if caso == TIPICO:
    linha = pd.Series({c: float(medianas[c]) for c in features})
    valor_orig, hora_orig = float(medianas["Amount"]), int(round(medianas["hora_relativa"]))
else:
    linha = exemplos.iloc[caso]
    valor_orig, hora_orig = float(linha["Amount"]), int(linha["hora_relativa"])

k_valor, k_hora = f"valor_{caso}", f"hora_{caso}"
colunas_v = [c for c in features if c.startswith("V")]
st.session_state.setdefault(k_valor, valor_orig)
st.session_state.setdefault(k_hora, hora_orig)
for c in colunas_v:
    st.session_state.setdefault(f"v_{caso}_{c}", float(linha[c]))


def voltar_original():
    st.session_state[k_valor] = valor_orig
    st.session_state[k_hora] = hora_orig
    for c in colunas_v:
        st.session_state[f"v_{caso}_{c}"] = float(linha[c])

st.subheader("2. Valor e horário")
c1, c2 = st.columns(2)
valor = c1.number_input(f"Valor da compra ({MOEDA})", min_value=0.0, max_value=valor_max, step=10.0,
                        format="%.2f", key=k_valor, help="Digite o valor e aperte Enter.")
hora = c2.slider("Horário (hora do dia, 0 a 23)", 0, 23, key=k_hora)

with st.expander("Avançado: editar as 28 características do perfil (V1 a V28)"):
    st.caption("Componentes anonimizados fornecidos pelo provedor da base, sem interpretação direta. "
               "Já vêm preenchidos com os valores do perfil escolhido.")
    grade = st.columns(4)
    for i, c in enumerate(colunas_v):
        with grade[i % 4]:
            st.number_input(c, step=0.1, format="%.8f", key=f"v_{caso}_{c}")

st.button("↺ Voltar aos valores originais", on_click=voltar_original)

linha_atual = linha.copy()
for c in colunas_v:
    linha_atual[c] = float(st.session_state[f"v_{caso}_{c}"])
perfil_alterado = not all(np.isclose(float(linha_atual[c]), float(linha[c]), rtol=0.0, atol=1e-8) for c in colunas_v)

proba = prever(linha_atual, valor, hora)
proba_orig = prever(linha, valor_orig, hora_orig)
eh_fraude = proba >= limiar

st.subheader("3. Resultado")
r1, r2 = st.columns(2)
r1.metric(
    "Pontuação de risco de fraude",
    f"{proba:.2%}",
    delta=f"{(proba - proba_orig) * 100:+.2f} pp vs. original",
    delta_color="off",
)
with r2:
    if eh_fraude:
        st.error("### 🚨 Possível fraude\nEncaminhar para revisão ou bloqueio.")
    else:
        st.success("### ✅ Parece legítima\nPode seguir normalmente.")
st.progress(min(max(proba, 0.0), 1.0))
st.caption(f"O alerta dispara a partir de {limiar:.0%}. Mexer em valor e horário muda a pontuação, "
           "mas pouco: o perfil (V1 a V28) é o que mais decide.")

if caso != TIPICO and st.toggle("Mostrar a resposta real"):
    real_fraude = int(linha["y_real"]) == 1
    st.write(f"Na vida real, esta transação **{'era fraude' if real_fraude else 'era legítima'}**.")
    if real_fraude == eh_fraude:
        st.success("O modelo acertou.")
    else:
        st.warning("O modelo errou neste caso. Nenhum modelo acerta tudo.")
    st.caption(f"Tipo de caso: {linha['rotulo']}")

with st.expander("Por que valor e horário mudam pouco o resultado?"):
    st.write(
        "O modelo decide principalmente pelas 28 características anonimizadas (V1 a V28). "
        "Elas ficam fixas em cada perfil. Valor e horário entram no cálculo, mas têm peso pequeno. "
        "Os gráficos mostram a pontuação deste perfil variando só um deles de cada vez."
    )
    aba_valor, aba_hora = st.tabs(["Variando o valor", "Variando o horário"])
    with aba_valor:
        montantes = np.geomspace(0.5, min(5000.0, valor_max), 30)
        curva = pd.DataFrame({"Pontuação de risco": [prever(linha_atual, m, hora) for m in montantes]},
                             index=pd.Index(np.round(montantes, 2), name=f"Valor ({MOEDA})"))
        st.line_chart(curva, height=220)
    with aba_hora:
        horas = list(range(24))
        curva_h = pd.DataFrame({"Pontuação de risco": [prever(linha_atual, valor, h) for h in horas]},
                               index=pd.Index(horas, name="Hora do dia"))
        st.line_chart(curva_h, height=220)

with st.expander("Verificação técnica: o app confere com o notebook?"):
    if caso != TIPICO and not perfil_alterado and abs(valor - valor_orig) < 1e-9 and hora == hora_orig:
        proba_nb = float(linha["proba_notebook"])
        diferenca = abs(proba - proba_nb)
        p1, p2, p3 = st.columns(3)
        p1.metric("No notebook", f"{proba_nb:.6f}")
        p2.metric("No aplicativo", f"{proba:.6f}")
        p3.metric("Diferença", f"{diferenca:.2e}")
        if diferenca < TOLERANCIA_PARIDADE:
            st.success("Paridade confirmada: o app dá a mesma previsão do notebook.")
        else:
            st.error("Divergência entre app e notebook. Verifique as versões das bibliotecas.")
    else:
        st.info("Escolha uma transação de exemplo (A, B, ...) e volte aos valores originais para comparar com o notebook.")
    entrada = pd.DataFrame([linha_atual[features].astype(float)], columns=features)
    entrada["Amount"], entrada["hora_relativa"] = float(valor), float(hora)
    st.caption("Valores enviados ao modelo:")
    st.dataframe(entrada.T.rename(columns={0: "valor"}))

with st.expander("Como funciona, em uma frase"):
    if algoritmo == "XGBoost":
        st.write(
            "Um XGBoost (muitas árvores de decisão em sequência, cada uma corrigindo os erros da anterior) "
            "aprendeu, com centenas de milhares de transações reais, quais combinações de características "
            "aparecem em fraudes e devolve uma pontuação de risco entre 0% e 100%. "
            "Só cerca de 0,17% das transações da base são fraude."
        )
    elif algoritmo == "Random Forest":
        st.write(
            "Um Random Forest (muitas árvores de decisão votando) aprendeu, com centenas de milhares de "
            "transações reais, quais combinações de características aparecem em fraudes, e devolve a fração "
            "de árvores que votam 'fraude'. Só cerca de 0,17% das transações da base são fraude."
        )
    else:
        st.write(
            f"O modelo ({meta['modelo_nome']}) aprendeu, com centenas de milhares de transações reais, "
            "quais combinações de características aparecem em fraudes. Só cerca de 0,17% das transações "
            "da base são fraude."
        )