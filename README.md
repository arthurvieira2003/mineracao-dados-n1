# Mineração de Dados - Aplicando CRISP-DM

Este repositório contém exemplos práticos de mineração de dados usando o processo CRISP-DM (Cross Industry Standard Process for Data Mining), desenvolvidos como parte de um trabalho acadêmico.

## Sobre o Projeto

Este projeto demonstra técnicas de mineração de dados através de exemplos práticos em Python, seguindo a metodologia CRISP-DM. Os exemplos cobrem desde a formulação de questões analíticas até a implementação de diferentes técnicas de modelagem.

## Arquivos do Projeto

1. **1_tecnicas_mineracao.py**: Demonstra cinco técnicas fundamentais de mineração de dados:

   - Classificação (previsão de inadimplência bancária)
   - Regressão (previsão de preços de imóveis)
   - Clusterização (segmentação de clientes)
   - Regras de Associação (análise de cesta de compras)
   - Detecção de Anomalias (identificação de transações fraudulentas)

2. **3_entendimento_preparacao_dados.py**: Aplica as etapas de Entendimento e Preparação de Dados do CRISP-DM ao problema de e-commerce.
   - Geração de dados brutos simulados usando Faker
   - Análise exploratória dos dados
   - Preparação dos dados para modelagem

## Estrutura de Diretórios

```
.
└── exemplos_crisp_dm/
    ├── 1_tecnicas_mineracao.py
    ├── 3_entendimento_preparacao_dados.py
    ├── imagens/                           # Visualizações geradas pelos scripts
    │   ├── anomalias_transacoes.png
    │   ├── avaliacao_vs_atraso.png
    │   └── distribuicao_atrasos.png
    └── dados_brutos_*.csv                 # Arquivos CSV gerados com dados sintéticos
```

## Requisitos

Os exemplos requerem Python 3.8+ e as seguintes bibliotecas:

```
pandas==2.0.0+
numpy==1.24.0+
matplotlib==3.7.0+
seaborn==0.12.0+
scikit-learn==1.2.0+
faker==18.0.0+
mlxtend==0.22.0+
```

Para instalar todas as dependências:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn faker mlxtend
```

## Execução dos Exemplos

Navegue até o diretório do projeto e execute cada script individualmente:

```bash
# No Windows
cd exemplos_crisp_dm
python 1_tecnicas_mineracao.py
python 3_entendimento_preparacao_dados.py

# No Linux/MacOS
cd exemplos_crisp_dm
python 1_tecnicas_mineracao.py
python 3_entendimento_preparacao_dados.py
```

## Metodologia CRISP-DM

O CRISP-DM (Cross Industry Standard Process for Data Mining) é uma metodologia para projetos de mineração de dados dividida em seis fases:

1. **Entendimento do Negócio**: Definição clara dos objetivos e requisitos do projeto.
2. **Entendimento dos Dados**: Análise exploratória para entender a estrutura e qualidade dos dados.
3. **Preparação dos Dados**: Limpeza, transformação e engenharia de características.
4. **Modelagem**: Aplicação de técnicas como classificação, regressão, clusterização, etc.
5. **Avaliação**: Validação dos resultados e avaliação da eficácia dos modelos.
6. **Implementação**: Colocação dos modelos em uso prático.

Cada exemplo neste repositório segue essas fases, demonstrando sua aplicação em diferentes técnicas de mineração de dados.

## Resultados Esperados

Ao executar os scripts, você obterá:

1. **Do script de técnicas de mineração**: Modelos treinados e resultados de avaliação para cada técnica, além de uma visualização de anomalias.
2. **Do script de entendimento e preparação**: Dados sintéticos gerados, análises exploratórias, visualizações e um conjunto de dados final preparado para modelagem.

## Solução de Problemas Comuns

### Erro ao Salvar Imagens

Se ocorrer um erro `FileNotFoundError` ao tentar salvar imagens, verifique se o diretório `imagens` existe:

```bash
# Criar diretório de imagens se não existir
cd exemplos_crisp_dm
mkdir -p imagens
```

### Erro com Biblioteca mlxtend

Se você encontrar o erro `ModuleNotFoundError: No module named 'mlxtend'`, instale a biblioteca usando:

```bash
pip install mlxtend
```

### Caminhos de Arquivo no Windows

Em sistemas Windows, os caminhos de arquivo nos scripts usam barras invertidas (`\`). Se você estiver executando em Linux/MacOS, pode ser necessário modificar os caminhos para usar barras normais (`/`).
