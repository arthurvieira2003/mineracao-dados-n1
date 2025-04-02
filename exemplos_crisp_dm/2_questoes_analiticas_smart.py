"""
Exemplo de problema com Questões Analíticas usando SMART
para o processo CRISP-DM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# EXEMPLO DE PROBLEMA: OTIMIZAÇÃO DE OPERAÇÕES LOGÍSTICAS EM E-COMMERCE
# -----------------------------------------------------------------------------
print("PROBLEMA DE NEGÓCIO: OTIMIZAÇÃO DE OPERAÇÕES LOGÍSTICAS EM E-COMMERCE")
print("=" * 80)
print("""
Contexto: Uma empresa de e-commerce está enfrentando problemas com atrasos nas entregas,
afetando a satisfação dos clientes e aumentando custos operacionais. A empresa deseja 
utilizar mineração de dados para otimizar suas operações logísticas, reduzir atrasos
e melhorar a experiência do cliente.

A empresa possui dados históricos de vendas, entregas, fornecedores, rotas de entrega,
tempo médio de processamento de pedidos e feedback dos clientes. Ela precisa identificar
gargalos, prever atrasos e otimizar seus processos logísticos.
""")

print("\nQUESTÕES ANALÍTICAS (QA) USANDO MÉTODO SMART")
print("=" * 80)

# QA 1: Previsão de Atrasos
print("\nQA 1: Podemos prever com 85% de precisão quais pedidos têm alta probabilidade de atraso,")
print("     considerando dados históricos dos últimos 12 meses, até o final do próximo trimestre?")
print("\nAnálise SMART:")
print("  - Específica (Specific): Identifica o objetivo claro de prever atrasos com um nível de precisão definido")
print("  - Mensurável (Measurable): Define 85% como meta de precisão do modelo preditivo")
print("  - Atingível (Attainable): Utiliza dados históricos disponíveis dos últimos 12 meses")
print("  - Relevante (Relevant): Diretamente relacionado ao problema de atrasos nas entregas")
print("  - Temporal (Time-related): Estabelece prazo para implementação até o final do próximo trimestre")

# QA 2: Identificação de Fatores de Atraso
print("\nQA 2: Quais são os 5 principais fatores que contribuem para atrasos nas entregas,")
print("     com base na análise estatística dos dados de entregas dos últimos 6 meses,")
print("     identificados em ordem de impacto no prazo de 2 meses?")
print("\nAnálise SMART:")
print("  - Específica (Specific): Foca nos principais fatores de atraso em ordem de impacto")
print("  - Mensurável (Measurable): Define a quantidade exata (5) de fatores a serem identificados")
print("  - Atingível (Attainable): Utiliza dados de entregas disponíveis dos últimos 6 meses")
print("  - Relevante (Relevant): Identifica causas-raiz que podem ser abordadas para melhorias")
print("  - Temporal (Time-related): Estabelece prazo de 2 meses para a análise")

# QA 3: Otimização de Rotas
print("\nQA 3: Como podemos reduzir em pelo menos 15% o tempo médio de entrega nas 10 rotas")
print("     mais congestionadas, implementando um algoritmo de otimização de rotas,")
print("     dentro de 3 meses?")
print("\nAnálise SMART:")
print("  - Específica (Specific): Foca nas 10 rotas mais problemáticas com um objetivo claro de otimização")
print("  - Mensurável (Measurable): Define meta de redução de 15% no tempo médio de entrega")
print("  - Atingível (Attainable): Considera apenas as 10 rotas principais, tornando o projeto viável")
print("  - Relevante (Relevant): Diretamente ligado à melhoria das operações logísticas")
print("  - Temporal (Time-related): Define prazo de 3 meses para implementação")

# QA 4: Segmentação de Clientes para Entrega Prioritária
print("\nQA 4: Qual segmentação de clientes em 3-5 grupos distintos, baseada no valor")
print("     de compra e frequência, permitiria implementar um sistema de priorização")
print("     de entregas que aumente a satisfação do cliente em 20% em 4 meses?")
print("\nAnálise SMART:")
print("  - Específica (Specific): Define a criação de segmentos de clientes para priorização de entregas")
print("  - Mensurável (Measurable): Define meta de aumento de 20% na satisfação do cliente")
print("  - Atingível (Attainable): Limita a segmentação a 3-5 grupos gerenciáveis")
print("  - Relevante (Relevant): Conecta-se à estratégia de priorização para melhor alocação de recursos")
print("  - Temporal (Time-related): Estabelece prazo de 4 meses para implementação e medição de resultados")

# QA 5: Previsão de Demanda por Região
print("\nQA 5: Podemos implementar um modelo preditivo que forecaste a demanda diária por região")
print("     com margem de erro máxima de 10%, utilizando dados históricos de 24 meses,")
print("     em um período de implementação de 3 meses?")
print("\nAnálise SMART:")
print("  - Específica (Specific): Foca na previsão de demanda por região com granularidade diária")
print("  - Mensurável (Measurable): Define meta de precisão com margem de erro máxima de 10%")
print("  - Atingível (Attainable): Utiliza dados históricos disponíveis de 24 meses")
print("  - Relevante (Relevant): Permite melhor planejamento da capacidade logística")
print("  - Temporal (Time-related): Estabelece prazo de 3 meses para implementação")

print("\n\nDESENVOLVIMENTO DO PROCESSO CRISP-DM")
print("=" * 80)
print("""
As Questões Analíticas SMART guiarão cada fase do processo CRISP-DM:

1. Entendimento do Negócio:
   - As QAs definem claramente os objetivos que vão orientar todo o projeto
   - Estabelecem critérios mensuráveis de sucesso (ex: redução de 15% no tempo de entrega)

2. Entendimento dos Dados:
   - As QAs indicam quais dados precisamos coletar (entregas, rotas, clientes, etc.)
   - Definem o período de dados a ser analisado (6-24 meses dependendo da questão)

3. Preparação dos Dados:
   - Guiam a seleção e transformação das variáveis relevantes para cada objetivo
   - Orientam a criação de features específicas (ex: congestionamento de rotas)

4. Modelagem:
   - Indicam os algoritmos apropriados para cada questão (classificação para atrasos, 
     clusterização para segmentação de clientes, etc.)

5. Avaliação:
   - Fornecem métricas claras para avaliar os resultados (85% de precisão, 20% de aumento
     na satisfação, etc.)

6. Implementação:
   - Definem prazos realistas para colocar as soluções em produção
   - Orientam a criação de produtos de dados específicos (dashboard de previsão, sistema
     de alerta de atrasos, etc.)
""") 