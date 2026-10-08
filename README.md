# Checkpoint 5 — Random Forest, XGBoost e LightGBM: Detecção de Fraude em Cartão de Crédito

**Disciplina**: Data Science & Statistical Computing (FIAP, 2026)  
**Professor**: Jones Egydio  
**Integrantes**: Bernardo Moreira - RM: 564103 / Pedro Batista - RM: 563220 / Larissa Shiba - RM: 560462 / Renan Jordão - RM: 560618

## Links

- **GitHub**: https://github.com/PedroH-Batista/Checkpoint-Data-Science.git
- **Aplicação Streamlit**: 

## Objetivo

Construir um projeto completo de classificação supervisionada, comparando Random Forest, XGBoost e LightGBM sob um mesmo protocolo experimental, com tuning por Grid Search e Optuna, análise de overfitting e underfitting, avaliação final em dados isolados e aplicação interativa em Streamlit integrada ao pipeline final.

## Problema

Dadas as características de uma transação de cartão de crédito, estimar a probabilidade de ela ser fraudulenta, para que apenas as transações mais suspeitas sejam encaminhadas à revisão.

- **Variável resposta**: `Class` (1 = fraude, 0 = legítima)
- **Tipo**: classificação binária com desbalanceamento extremo (cerca de 0,17% de fraudes)
- **Métrica principal**: PR-AUC (Average Precision). A acurácia é inadequada porque um modelo que sempre prevê "legítima" acerta mais de 99,8% dos casos.
- **Métricas auxiliares**: ROC-AUC, Precision, Recall, F1 e MCC

## Base de Dados

- **Fonte**: "Credit Card Fraud Detection" (Machine Learning Group, ULB, em parceria com a Worldline; Kaggle)
- **Licença**: Open Database License (ODbL), conforme a página do dataset no Kaggle
- **Mirror usado pelo notebook**: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
- **Tamanho**: 284.807 transações, 31 colunas, 492 fraudes
- **Unidade observacional**: uma transação de cartão (portadores europeus, dois dias de setembro de 2013)
- **Variáveis**: `Time`, `V1` a `V28` (componentes principais anonimizados), `Amount` e `Class`

**Obtenção da base**: o notebook baixa o arquivo automaticamente para `dados/base_bruta.csv`. A base bruta (cerca de 144 MB) e a base tratada excedem o limite de 100 MB por arquivo do GitHub e por isso **não são versionadas** (ver `.gitignore`). Para obtê-la manualmente, baixe o arquivo `creditcard.csv` pelo mirror acima ou pelo Kaggle e salve como `dados/base_bruta.csv`. O aplicativo Streamlit não depende da base.

## Estrutura do Projeto

```
Checkpoint-05-Fraude/
├── app.py                           # App Streamlit para predição interativa
├── notebook.ipynb                   # Notebook completo (EDA, modelagem, tuning, avaliação)
├── requirements.txt                 # Dependências do projeto
├── README.md
├── .gitignore
├── dados/
│   ├── base_bruta.csv               # Gerado pelo notebook (não versionado)
│   └── base_tratada.csv             # Gerado pelo notebook (não versionado)
└── modelo/
    ├── modelo.pkl                   # Pipeline completo (imputador + modelo final)
    ├── metadados.json               # Features, ranges, limiar, métricas, hiperparâmetros
    └── exemplos_consistencia.csv    # Observações do teste de consistência (paridade com o app)
```

## Protocolo Experimental

| Item | Decisão |
|------|---------|
| Split | 80% treino e 20% teste, `stratify=y`, `random_state=42` |
| Validação cruzada | `StratifiedKFold` com 5 folds, `shuffle=True`, `random_state=42`, somente dentro do treino |
| Métrica principal | PR-AUC (Average Precision) |
| Pré-processamento | `SimpleImputer(median)` dentro do `Pipeline` (ajustado apenas nos folds de treino) |
| Limiar de decisão | Máximo F1 sobre previsões out-of-fold do treino (o teste não participa) |
| Conjunto de teste | Usado uma única vez, na avaliação final do modelo escolhido |

Os três algoritmos usam o mesmo teste, os mesmos folds, a mesma métrica e o mesmo pipeline.

## Wrangling Realizado

1. Diagnóstico: tipos, ausências, duplicidades, valores impossíveis
2. Remoção das duplicatas exatas antes do split (evita o mesmo registro em treino e teste)
3. Derivação de `hora_relativa = (Time mod 86400) // 3600` e exclusão de `Time`
4. Nenhum outlier removido: valores extremos de `Amount` podem ser o próprio padrão de fraude
5. Verificação final de ausências e duplicidades na base tratada

**Análise de data leakage**: a base não possui identificadores nem campos posteriores ao evento. As variáveis `V1` a `V28` foram geradas por PCA não supervisionado pelo provedor sobre a base inteira, uma limitação inevitável que está registrada no notebook.

## Modelos e Tuning

| Etapa | Descrição |
|-------|-----------|
| Baseline | Random Forest, XGBoost e LightGBM com hiperparâmetros de referência, sem tratamento de desbalanceamento |
| Grid Search | Grade de 8 combinações por algoritmo (complexidade, número de árvores e taxa de aprendizado), 5 folds |
| Optuna | TPE com 20 trials por algoritmo, mesma métrica e mesmos folds; espaço inclui regularização, amostragem e tratamento de desbalanceamento |
| Seleção | Nove configurações comparadas; regra do um desvio-padrão com preferência por menor gap treino-validação e menor custo |
| Diagnóstico | Gap treino-validação, curva de aprendizado, curva Precision-Recall e matriz de confusão |

## Resultados

Execute o notebook até o final: a última célula imprime as tabelas abaixo já em formato Markdown, prontas para colar.

### Comparação das nove configurações (validação cruzada no treino)

| Configuração | AP treino | AP CV (média) | AP CV (dp) | Gap | Tempo por fold (s) |
| --- | --- | --- | --- | --- | --- |
| Random Forest - Baseline **(escolhida)** | 0.9605 | 0.8394 | 0.0344 | 0.1211 | 6.8 |
| Random Forest - Grid Search | 0.9948 | 0.8436 | 0.0336 | 0.1512 | 15.7 |
| Random Forest - Optuna | 0.9792 | 0.8401 | 0.0336 | 0.1391 | 18.4 |
| XGBoost - Baseline | 1.0000 | 0.8403 | 0.0271 | 0.1597 | 1.4 |
| XGBoost - Grid Search | 1.0000 | 0.8465 | 0.0261 | 0.1535 | 1.8 |
| XGBoost - Optuna | 1.0000 | 0.8552 | 0.0300 | 0.1448 | 2.4 |
| LightGBM - Baseline | 0.2715 | 0.2694 | 0.1595 | 0.0022 | 1.1 |
| LightGBM - Grid Search | 0.8968 | 0.6648 | 0.0966 | 0.2320 | 1.8 |
| LightGBM - Optuna | 1.0000 | 0.8549 | 0.0308 | 0.1451 | 1.1 |

### Modelo final no conjunto de teste

| Métrica | Valor |
| --- | --- |
| PR-AUC (AP) | 0.7857 |
| ROC-AUC | 0.9672 |
| Precision | 0.9595 |
| Recall | 0.7474 |
| F1 | 0.8402 |
| MCC | 0.8466 |
| Limiar de decisão | 0.4124 |

**Modelo escolhido**: XGBoost - Optuna

## Como Executar

### Pré-requisitos

- Python 3.13+
- pip

### Instalação

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux ou macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Notebook

Abra `notebook.ipynb` no VS Code ou Jupyter e execute todas as células em sequência (Run All).  
Isso baixa a base, treina e compara os modelos e gera os arquivos em `dados/` e `modelo/`.

**Tempo estimado**: o Grid Search e o Optuna ajustam centenas de modelos sobre cerca de 227 mil transações de treino. Em um notebook comum, a execução completa pode levar de 1 a 3 horas, com a Random Forest sendo a etapa mais lenta. Para um teste rápido, reduza `N_FOLDS` e `N_TRIALS` na célula de configuração, lembrando que o enunciado exige no mínimo 20 trials por modelo na entrega final.

### App Streamlit

```bash
streamlit run app.py
```

O app carrega o pipeline salvo pelo notebook e permite:
- inserir manualmente hora, valor e componentes `V1` a `V28`;
- carregar casos do teste de consistência para reproduzir a previsão do notebook;
- ver o alerta de extrapolação quando uma entrada sai do intervalo observado no treino;
- conferir automaticamente a paridade entre a probabilidade do app e a do notebook.

### Teste de paridade notebook–interface

1. Execute o notebook completo (a seção 7.5 confirma a paridade com o pipeline carregado do disco).
2. Rode `streamlit run app.py` e selecione, em "Origem dos dados", um dos casos do teste de consistência.
3. A seção "Paridade com o notebook" mostra as duas probabilidades e a diferença absoluta.

### Deploy no Streamlit Community Cloud

1. Publique o repositório no GitHub, incluindo a pasta `modelo/` gerada pelo notebook. O arquivo `modelo.pkl` precisa ter menos de 100 MB.
2. No Streamlit Community Cloud, crie um novo app apontando para `app.py`.
3. Registre os links do GitHub e do Streamlit na célula 7.3 do notebook e no topo deste README.

## Tecnologias

- Python 3.13
- pandas, numpy, scikit-learn
- XGBoost, LightGBM, Optuna
- matplotlib, seaborn
- Streamlit

## Referências

- BREIMAN, Leo. Random Forests. *Machine Learning*, v. 45, p. 5–32, 2001.
- CHEN, Tianqi; GUESTRIN, Carlos. XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*, 2016.
- KE, Guolin et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*, 2017.
- AKIBA, Takuya et al. Optuna: A Next-generation Hyperparameter Optimization Framework. *ACM SIGKDD*, 2019.
- DAL POZZOLO, Andrea et al. Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE CIDM*, 2015.
