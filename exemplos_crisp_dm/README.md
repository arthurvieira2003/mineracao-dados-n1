# Exemplos de Mineração de Dados com CRISP-DM

Este repositório contém exemplos em Python que demonstram as técnicas de mineração de dados utilizando a metodologia CRISP-DM (Cross Industry Standard Process for Data Mining).

## Arquivos do Projeto

1. **1_tecnicas_mineracao.py**: Demonstra cinco técnicas fundamentais de mineração de dados:

   - Classificação (previsão de inadimplência bancária)
   - Regressão (previsão de preços de imóveis)
   - Clusterização (segmentação de clientes)
   - Regras de Associação (análise de cesta de compras)
   - Detecção de Anomalias (identificação de transações fraudulentas)

2. **2_questoes_analiticas_smart.py**: Aborda o uso de Questões Analíticas (QA) apoiadas pelo método SMART para um problema de otimização de operações logísticas em e-commerce.

   - Demonstra cinco questões analíticas bem formuladas
   - Explica como cada questão atende aos critérios SMART

3. **3_entendimento_preparacao_dados.py**: Aplica as etapas de Entendimento e Preparação de Dados do CRISP-DM ao problema de e-commerce.
   - Geração de dados brutos simulados
   - Análise exploratória dos dados
   - Preparação dos dados para modelagem

## Execução dos Exemplos

Cada exemplo pode ser executado independentemente. Para executar, certifique-se de ter as dependências necessárias instaladas:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn faker mlxtend
```

Para executar os exemplos, use:

```bash
python exemplos_crisp_dm/1_tecnicas_mineracao.py
python exemplos_crisp_dm/2_questoes_analiticas_smart.py
python exemplos_crisp_dm/3_entendimento_preparacao_dados.py
```

## Relação com CRISP-DM

Estes exemplos ilustram as principais fases do CRISP-DM:

1. **Entendimento do Negócio**: Definição clara dos objetivos e requisitos do projeto.
2. **Entendimento dos Dados**: Análise exploratória para entender a estrutura e qualidade dos dados.
3. **Preparação dos Dados**: Limpeza, transformação e engenharia de características.
4. **Modelagem**: Aplicação de técnicas como classificação, regressão, clusterização, etc.
5. **Avaliação**: Validação dos resultados e avaliação da eficácia dos modelos.
6. **Implementação**: Colocação dos modelos em uso prático.
