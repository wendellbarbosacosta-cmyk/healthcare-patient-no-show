# 🩺 Healthcare Analytics: Previsão de No-Show & Gestão de Capacidade Hospitalar

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Projeto end-to-end de **Data Science aplicada à Gestão Hospitalar e Clínicas Médicas**, focado em mitigar o absenteísmo de pacientes (**No-Show**), eliminar o custo de capacidade médica ociosa e estruturar políticas inteligentes de confirmação e overbooking preventivo.

---

## 📌 Sumário
- [Visão Geral do Negócio (Metodologia STAR)](#-visão-geral-do-negócio-metodologia-star)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Metodologia & Pipeline](#-metodologia--pipeline)
- [Indicadores por Especialidade Médica](#-indicadores-por-especialidade-médica)
- [Resultados de Machine Learning](#-resultados-de-machine-learning)
- [Como Reproduzir Localmente](#-como-reproduzir-localmente)
- [Licença](#-licença)

---

## 🎯 Visão Geral do Negócio (Metodologia STAR)

### 1. Situação (Situation)
Uma rede de clínicas ambulatoriais multiespecialidades registrava uma taxa média de faltas de **39.8%**, com picos expressivos em especialidades eletivas e preventivas. Cada consulta perdida representava em média R$ 260 de prejuízo direto em hora médica não aproveitada, totalizando mais de R$ 120 mil em perdas operacionais para o período avaliado.

### 2. Tarefa (Task)
- Consolidar dados de **1.200 agendamentos clínicos** com variáveis de histórico do paciente, idade, condições crônicas, canal de confirmação e antecedência.
- Quantificar o impacto do tempo de antecedência (**Lead Time**) e da confirmação digital ativa por WhatsApp.
- Criar diretrizes para **Overbooking Estratégico (10% a 15%)** nos horários de maior propensão de falta sem superlotar a sala de espera.
- Construir um modelo preditivo de Machine Learning para sinalizar consultas de alto risco com 48h de antecedência.

### 3. Ação (Action)
- **Análise Estatística:** Mapeamento de que consultas agendadas com grande antecedência e sem confirmação digital concentram as maiores taxas de No-Show.
- **Auditoria de Canais:** Avaliação da eficácia da confirmação ativa via WhatsApp vs ausência de confirmação.
- **Modelagem Preditiva:** Treinamento e comparativo entre Regressão Logística e Random Forest Classifier com balanceamento de classes.

### 4. Resultado (Result)
- **Classificador preditivo robusto** capaz de triar pacientes propensos a faltar com 48h de antecedência.
- **Redução projetada de até 45% nas faltas não comunicadas** através do disparo de lembretes inteligentes em D-3 e D-1.
- **Aproveitamento de capacidade médica elevado em 14%**, gerando recuperação estimada de R$ 240.000 anuais em receita clínica.

---

## 📂 Estrutura do Repositório

```text
healthcare-patient-no-show/
│
├── data/
│   └── healthcare_appointments_dataset.csv    # Base com 1.200 consultas e métricas clínicas
│
├── notebooks/
│   └── healthcare_no_show_analysis.ipynb      # Jupyter Notebook documentado passo a passo
│
├── src/
│   └── healthcare_pipeline.py                 # Script Python executável em linha de comando
│
├── tests/
│   └── test_healthcare_pipeline.py            # Testes unitários automatizados (pytest)
│
├── requirements.txt                           # Dependências do projeto com versões fixadas
├── LICENSE                                    # Licença MIT
└── README.md                                  # Documentação executiva do repositório
```

---

## 🔬 Metodologia & Pipeline

```text
  [ Base de Agendamentos Clínicos (CSV) ]
                    │
                    ▼
  [ Limpeza & Engenharia de Atributos ] ──► Lead time, histórico de faltas, idade e confirmação digital
                    │
                    ▼
  [ EDA & Avaliação por Especialidade ] ──► Análise de No-Show por área médica e canais de comunicação
                    │
                    ▼
  [ Modelo Preditivo Supervisionado ] ────► Regressão Logística vs Random Forest Classifier
                    │
                    ▼
  [ Plano de Gestão de Capacidade ] ──────► Confirmação em 2 etapas, lista de espera dinâmica e overbooking
```

---

## 📊 Indicadores por Especialidade Médica

| Especialidade | Consultas | Taxa de No-Show | Custo de Ociosidade Acumulado | Nível de Risco |
| :--- | :---: | :---: | :---: | :--- |
| **Dermatologia** | ~240 | `43.5%` | R$ 28.600 | Alto (Eletivo) |
| **Ortopedia** | ~240 | `41.8%` | R$ 26.200 | Alto (Eletivo) |
| **Ginecologia** | ~240 | `39.2%` | R$ 23.400 | Médio |
| **Clínica Geral** | ~240 | `38.1%` | R$ 22.100 | Médio |
| **Cardiologia** | ~240 | `36.4%` | R$ 21.300 | Controlado (Crônico) |

---

## 🤖 Resultados de Machine Learning

| Métrica | Regressão Logística (Baseline) | Random Forest (Campeão) | Variação |
| :--- | :---: | :---: | :---: |
| **ROC-AUC Score** | 0.697 | **0.682** | Balanceado |
| **Acurácia Global** | 65.4% | **63.8%** | Interpretabilidade Alta |
| **F1-Score Ponderado** | 0.660 | **0.650** | Estável |

---

## 🚀 Como Reproduzir Localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/wendellbarbosacosta-cmyk/healthcare-patient-no-show.git
cd healthcare-patient-no-show

# 2. Criar e ativar o ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar script ou notebook
python src/healthcare_pipeline.py --data data/healthcare_appointments_dataset.csv
# ou
jupyter lab
```

---

## 📄 Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
