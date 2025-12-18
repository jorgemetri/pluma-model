# 🪶 Pluma Model Project

Este repositório contém o pipeline de Machine Learning e a aplicação Streamlit para o projeto **Pluma**. A solução foi desenvolvida para ser executada nativamente no ambiente **Databricks**, utilizando Databricks Repos (Git Folders).

## 📂 Estrutura do Projeto

A organização das pastas segue uma lógica modular para separar a análise exploratória, engenharia de features, treinamento e a aplicação final.

```text
pluma-model/
├── app/                  # Código da aplicação Streamlit (Front-end)
│   └── app.py            # Ponto de entrada do Streamlit
├── conf/                 # Arquivos de configuração (Parâmetros, Caminhos)
│   └── settings.yaml     # Configurações globais (não comitar segredos!)
├── notebooks/            # Notebooks Databricks numerados sequencialmente
│   ├── 00_eda_inicial    # Análise Exploratória de Dados
│   ├── 01_feature_eng    # Criação e processamento de features
│   └── 02_train_model    # Treinamento e registro do modelo no MLflow
├── src/                  # Código fonte compartilhado (Bibliotecas locais)
│   ├── __init__.py
│   ├── preprocessing.py  # Funções de limpeza usadas no treino e no app
│   └── utils.py          # Funções auxiliares (ex: carregar configs)
├── README.md             # Documentação do projeto
└── requirements.txt      # Dependências Python (pip)
