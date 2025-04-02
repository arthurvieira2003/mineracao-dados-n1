"""
Exemplos de técnicas de mineração de dados usando CRISP-DM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------
# 1. CLASSIFICAÇÃO - Previsão de inadimplência de clientes bancários
# ------------------------------------------------------------------------
print("Exemplo 1: CLASSIFICAÇÃO")

# CRISP-DM: Entendimento do Negócio
# Problema: Banco quer prever quais clientes têm maior risco de inadimplência

# CRISP-DM: Entendimento dos Dados
# Simulando dados de clientes bancários
np.random.seed(42)
n_samples = 1000

# Gerando características dos clientes (idade, renda, histórico, etc.)
idade = np.random.randint(18, 80, n_samples)
renda = np.random.randint(1000, 15000, n_samples)
tempo_emprego = np.random.randint(0, 40, n_samples)
historico_credito = np.random.randint(300, 850, n_samples)

# CRISP-DM: Preparação dos Dados
# Criando um DataFrame
dados_clientes = pd.DataFrame({
    'idade': idade,
    'renda': renda,
    'tempo_emprego': tempo_emprego,
    'historico_credito': historico_credito
})

# Definindo a variável alvo (1: inadimplente, 0: adimplente)
dados_clientes['inadimplente'] = (
    (dados_clientes['historico_credito'] < 600) & 
    ((dados_clientes['renda'] < 3000) | (dados_clientes['tempo_emprego'] < 2))
).astype(int)

# CRISP-DM: Modelagem
from sklearn.ensemble import RandomForestClassifier

X = dados_clientes.drop('inadimplente', axis=1)
y = dados_clientes['inadimplente']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinando o modelo
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train_scaled, y_train)

# CRISP-DM: Avaliação
from sklearn.metrics import classification_report, accuracy_score

y_pred = modelo_rf.predict(X_test_scaled)
print(f"Acurácia do modelo: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# CRISP-DM: Implementação
# Exemplo de como o modelo seria usado para novos clientes
novo_cliente = pd.DataFrame({
    'idade': [35],
    'renda': [2500],
    'tempo_emprego': [1],
    'historico_credito': [580]
})

novo_cliente_scaled = scaler.transform(novo_cliente)
predicao = modelo_rf.predict(novo_cliente_scaled)[0]
print(f"\nPrevisão para novo cliente: {'Inadimplente' if predicao == 1 else 'Adimplente'}")

# ------------------------------------------------------------------------
# 2. REGRESSÃO - Previsão de preços de imóveis
# ------------------------------------------------------------------------
print("\n\nExemplo 2: REGRESSÃO")

# CRISP-DM: Entendimento do Negócio
# Problema: Imobiliária quer prever preços de casas para otimizar avaliações

# CRISP-DM: Entendimento dos Dados
# Simulando dados de imóveis
n_samples = 1000
area = np.random.randint(50, 300, n_samples)
quartos = np.random.randint(1, 6, n_samples)
idade_imovel = np.random.randint(0, 50, n_samples)
distancia_centro = np.random.randint(1, 30, n_samples)

# CRISP-DM: Preparação dos Dados
dados_imoveis = pd.DataFrame({
    'area': area,
    'quartos': quartos,
    'idade_imovel': idade_imovel,
    'distancia_centro': distancia_centro
})

# Criando preço baseado nas características (com ruído)
dados_imoveis['preco'] = (
    2000 * dados_imoveis['area'] + 
    50000 * dados_imoveis['quartos'] - 
    5000 * dados_imoveis['idade_imovel'] - 
    10000 * dados_imoveis['distancia_centro'] + 
    np.random.normal(0, 50000, n_samples)
)

# CRISP-DM: Modelagem
from sklearn.linear_model import LinearRegression

X = dados_imoveis.drop('preco', axis=1)
y = dados_imoveis['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_regr = LinearRegression()
modelo_regr.fit(X_train, y_train)

# CRISP-DM: Avaliação
from sklearn.metrics import mean_squared_error, r2_score

y_pred = modelo_regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio: {mse:.2f}")
print(f"R² Score: {r2:.4f}")
print("\nCoeficientes do modelo:")
for feature, coef in zip(X.columns, modelo_regr.coef_):
    print(f"- {feature}: {coef:.2f}")

# ------------------------------------------------------------------------
# 3. CLUSTERING - Segmentação de clientes para marketing
# ------------------------------------------------------------------------
print("\n\nExemplo 3: CLUSTERIZAÇÃO")

# CRISP-DM: Entendimento do Negócio
# Problema: Empresa de varejo quer segmentar clientes para campanhas de marketing

# CRISP-DM: Entendimento dos Dados
# Simulando dados de consumo de clientes
n_samples = 500
recencia = np.random.randint(1, 365, n_samples)  # dias desde última compra
frequencia = np.random.randint(1, 50, n_samples)  # número de compras no ano
valor_gasto = np.random.randint(100, 10000, n_samples)  # valor gasto no ano

# CRISP-DM: Preparação dos Dados
dados_clientes_rfm = pd.DataFrame({
    'recencia': recencia,
    'frequencia': frequencia,
    'valor_gasto': valor_gasto
})

# Normalizando os dados
X = dados_clientes_rfm.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CRISP-DM: Modelagem
from sklearn.cluster import KMeans

# Determinando o número ideal de clusters com o método do cotovelo
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Escolhendo k=4 clusters após análise do cotovelo
modelo_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = modelo_kmeans.fit_predict(X_scaled)
dados_clientes_rfm['cluster'] = clusters

# CRISP-DM: Avaliação e Interpretação
print("Perfil dos clusters:")
for i in range(4):
    print(f"Cluster {i}:")
    cluster_data = dados_clientes_rfm[dados_clientes_rfm['cluster'] == i]
    print(f"  - Tamanho: {len(cluster_data)} clientes")
    print(f"  - Recência média: {cluster_data['recencia'].mean():.1f} dias")
    print(f"  - Frequência média: {cluster_data['frequencia'].mean():.1f} compras")
    print(f"  - Valor gasto médio: R$ {cluster_data['valor_gasto'].mean():.2f}")
    print()

# CRISP-DM: Implementação
# Exemplo de estratégias para cada cluster
estrategias = [
    "Oferecer descontos para reativação",
    "Programa de fidelidade com bônus",
    "Upsell de produtos premium",
    "Manter engajamento com comunicação frequente"
]

print("Estratégias recomendadas por cluster:")
for i in range(4):
    print(f"Cluster {i}: {estrategias[i]}")

# ------------------------------------------------------------------------
# 4. REGRAS DE ASSOCIAÇÃO - Análise de cesta de compras
# ------------------------------------------------------------------------
print("\n\nExemplo 4: REGRAS DE ASSOCIAÇÃO")

# CRISP-DM: Entendimento do Negócio
# Problema: Supermercado quer entender quais produtos são comprados juntos

# CRISP-DM: Entendimento dos Dados
# Criando transações simuladas
produtos = ['pão', 'leite', 'ovos', 'manteiga', 'queijo', 'iogurte', 
           'carne', 'frango', 'arroz', 'feijão', 'macarrão', 'molho']

transacoes = []
for _ in range(1000):
    n_itens = np.random.randint(1, 8)
    transacao = list(np.random.choice(produtos, size=n_itens, replace=False))
    # Adicionando algumas regras para criar padrões
    if 'pão' in transacao and np.random.random() < 0.7:
        transacao.append('manteiga')
    if 'arroz' in transacao and np.random.random() < 0.6:
        transacao.append('feijão')
    transacoes.append(transacao)

# CRISP-DM: Preparação dos Dados
# Convertendo para formato adequado para o algoritmo apriori
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit_transform(transacoes)
df_transacoes = pd.DataFrame(te_array, columns=te.columns_)

# CRISP-DM: Modelagem
from mlxtend.frequent_patterns import apriori, association_rules

# Encontrando conjuntos frequentes
itemsets_freq = apriori(df_transacoes, min_support=0.05, use_colnames=True)

# Gerando regras de associação
regras = association_rules(itemsets_freq, metric="confidence", min_threshold=0.5)

# CRISP-DM: Avaliação
print("Top 5 regras de associação por confiança:")
regras_ordenadas = regras.sort_values('confidence', ascending=False).head(5)
for idx, row in regras_ordenadas.iterrows():
    antecedentes = ', '.join(list(row['antecedents']))
    consequentes = ', '.join(list(row['consequents']))
    print(f"{antecedentes} → {consequentes}")
    print(f"  - Suporte: {row['support']:.4f}")
    print(f"  - Confiança: {row['confidence']:.4f}")
    print(f"  - Lift: {row['lift']:.4f}")
    print()

# ------------------------------------------------------------------------
# 5. DETECÇÃO DE ANOMALIAS - Identificação de transações fraudulentas
# ------------------------------------------------------------------------
print("\n\nExemplo 5: DETECÇÃO DE ANOMALIAS")

# CRISP-DM: Entendimento do Negócio
# Problema: Banco quer identificar transações possivelmente fraudulentas

# CRISP-DM: Entendimento dos Dados
# Simulando dados de transações bancárias
n_samples = 1000
valor = np.concatenate([
    np.random.normal(500, 300, n_samples - 50),  # Transações normais
    np.random.normal(5000, 2000, 50)            # Transações anômalas (potencialmente fraudulentas)
])
hora_do_dia = np.concatenate([
    np.random.normal(12, 5, n_samples - 50),    # Horário normal
    np.random.normal(3, 2, 50)                  # Horário noturno (suspeito)
])
# Limitando hora do dia entre 0 e 24
hora_do_dia = np.clip(hora_do_dia, 0, 24)

# CRISP-DM: Preparação dos Dados
dados_transacoes = pd.DataFrame({
    'valor': valor,
    'hora_do_dia': hora_do_dia
})

X = dados_transacoes.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CRISP-DM: Modelagem
from sklearn.ensemble import IsolationForest

modelo_if = IsolationForest(contamination=0.05, random_state=42)
dados_transacoes['anomalia'] = modelo_if.fit_predict(X_scaled)
dados_transacoes['anomalia'] = dados_transacoes['anomalia'].map({1: 0, -1: 1})  # 1 = anomalia, 0 = normal

# CRISP-DM: Avaliação
print(f"Total de transações: {len(dados_transacoes)}")
print(f"Transações identificadas como anômalas: {dados_transacoes['anomalia'].sum()}")

# Estatísticas para transações normais vs. anômalas
print("\nEstatísticas para transações normais:")
normal = dados_transacoes[dados_transacoes['anomalia'] == 0]
print(f"  - Valor médio: R$ {normal['valor'].mean():.2f}")
print(f"  - Hora média: {normal['hora_do_dia'].mean():.2f}")

print("\nEstatísticas para transações anômalas:")
anomalas = dados_transacoes[dados_transacoes['anomalia'] == 1]
print(f"  - Valor médio: R$ {anomalas['valor'].mean():.2f}")
print(f"  - Hora média: {anomalas['hora_do_dia'].mean():.2f}")

# Visualização
plt.figure(figsize=(10, 6))
plt.scatter(
    dados_transacoes['valor'],
    dados_transacoes['hora_do_dia'],
    c=dados_transacoes['anomalia'],
    cmap='coolwarm',
    alpha=0.6
)
plt.colorbar(label='Anomalia')
plt.xlabel('Valor da Transação (R$)')
plt.ylabel('Hora do Dia')
plt.title('Detecção de Anomalias em Transações Bancárias')
plt.tight_layout()
plt.savefig('exemplos_crisp_dm/imagens/anomalias_transacoes.png') 