"""
Exemplo de Entendimento e Preparação de Dados usando CRISP-DM
baseado no problema de E-commerce da questão anterior
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from datetime import datetime, timedelta
import random

# Definindo um seed para reprodutibilidade
np.random.seed(42)
random.seed(42)
fake = Faker('pt_BR')
Faker.seed(42)

print("ENTENDIMENTO E PREPARAÇÃO DE DADOS - E-COMMERCE LOGÍSTICA")
print("=" * 80)
print("""
Continuando com o problema de negócio apresentado anteriormente:
Otimização de operações logísticas de uma empresa de e-commerce para reduzir atrasos
nas entregas e melhorar a satisfação dos clientes.

Vamos demonstrar as etapas de Entendimento e Preparação de Dados do CRISP-DM.
""")

# ----------------------------------------------------------------------
# GERAÇÃO DE DADOS BRUTOS
# ----------------------------------------------------------------------
print("\n1. GERAÇÃO DE DADOS BRUTOS")
print("-" * 50)

# Definindo parâmetros para geração de dados
n_pedidos = 1000
n_clientes = 300
n_produtos = 150
n_fornecedores = 10
n_regioes = 5
n_transportadoras = 8

# Período dos dados
data_inicio = datetime(2023, 1, 1)
data_fim = datetime(2023, 12, 31)
dias_periodo = (data_fim - data_inicio).days

print(f"Gerando conjunto de dados com {n_pedidos} pedidos no período de {data_inicio.date()} a {data_fim.date()}")

# Função para gerar data aleatória no período
def gerar_data_aleatoria():
    return data_inicio + timedelta(days=random.randint(0, dias_periodo))

# Gerando dados de clientes
clientes = []
for i in range(n_clientes):
    cliente = {
        'cliente_id': i + 1,
        'nome': fake.name(),
        'email': fake.email(),
        'telefone': fake.phone_number(),
        'endereco': fake.address(),
        'cidade': fake.city(),
        'estado': fake.state_abbr(),
        'regiao_id': random.randint(1, n_regioes),
        'data_cadastro': gerar_data_aleatoria()
    }
    clientes.append(cliente)

df_clientes = pd.DataFrame(clientes)

# Gerando dados de produtos
categorias = ['Eletrônicos', 'Vestuário', 'Alimentos', 'Casa e Decoração', 
              'Esportes', 'Beleza', 'Livros', 'Brinquedos', 'Pet Shop', 'Saúde']

produtos = []
for i in range(n_produtos):
    produto = {
        'produto_id': i + 1,
        'nome': f"Produto_{i+1}",
        'categoria': random.choice(categorias),
        'preco': round(random.uniform(20, 2000), 2),
        'peso_kg': round(random.uniform(0.1, 30), 2),
        'fornecedor_id': random.randint(1, n_fornecedores),
        'estoque': random.randint(0, 500),
        'tempo_processamento_dias': random.randint(1, 5)
    }
    produtos.append(produto)

df_produtos = pd.DataFrame(produtos)

# Gerando dados de transportadoras
transportadoras = []
for i in range(n_transportadoras):
    trans = {
        'transportadora_id': i + 1,
        'nome': f"Transportadora_{chr(65+i)}",
        'tempo_medio_entrega_dias': random.randint(2, 10),
        'taxa_atraso': round(random.uniform(0.05, 0.3), 2),
        'custo_por_km': round(random.uniform(0.5, 2.5), 2)
    }
    transportadoras.append(trans)

df_transportadoras = pd.DataFrame(transportadoras)

# Gerando dados de regiões
regioes = []
for i in range(n_regioes):
    regiao = {
        'regiao_id': i + 1,
        'nome': f"Região_{chr(65+i)}",
        'distancia_media_km': random.randint(50, 1500),
        'tempo_medio_entrega_dias': random.randint(2, 15),
        'taxa_congestionamento': round(random.uniform(1.0, 2.5), 2)
    }
    regioes.append(regiao)

df_regioes = pd.DataFrame(regioes)

# Gerando dados de pedidos e entregas
pedidos = []
entregas = []

status_opcoes = ['Entregue', 'Em trânsito', 'Processando', 'Cancelado']
metodo_pagamento = ['Cartão de Crédito', 'Boleto', 'Pix', 'Transferência']

for i in range(n_pedidos):
    # Dados do pedido
    cliente_id = random.randint(1, n_clientes)
    cliente = next(c for c in clientes if c['cliente_id'] == cliente_id)
    regiao_id = cliente['regiao_id']
    data_pedido = gerar_data_aleatoria()
    
    # Calculando valor do pedido
    n_itens = random.randint(1, 5)
    produtos_pedido = random.sample(range(1, n_produtos + 1), n_itens)
    valor_total = sum(next(p['preco'] for p in produtos if p['produto_id'] == pid) for pid in produtos_pedido)
    
    # Processamento do pedido
    tempo_processamento = random.randint(1, 3)
    
    # Escolhendo transportadora
    transportadora_id = random.randint(1, n_transportadoras)
    transportadora = next(t for t in transportadoras if t['transportadora_id'] == transportadora_id)
    
    # Calculando tempo estimado de entrega
    regiao = next(r for r in regioes if r['regiao_id'] == regiao_id)
    tempo_base_entrega = transportadora['tempo_medio_entrega_dias']
    fator_regional = regiao['taxa_congestionamento']
    tempo_estimado_entrega = int(tempo_base_entrega * fator_regional)
    
    # Adicionando variabilidade para simular atrasos
    probabilidade_atraso = transportadora['taxa_atraso']
    atraso = 0
    if random.random() < probabilidade_atraso:
        atraso = random.randint(1, 7)  # 1 a 7 dias de atraso
    
    # Calculando datas
    data_envio = data_pedido + timedelta(days=tempo_processamento)
    data_entrega_prevista = data_envio + timedelta(days=tempo_estimado_entrega)
    data_entrega_real = data_entrega_prevista + timedelta(days=atraso)
    
    # Determinando status do pedido
    if data_entrega_real <= datetime.now():
        status = 'Entregue'
    elif data_envio <= datetime.now():
        status = 'Em trânsito'
    else:
        status = 'Processando'
    
    # Avaliação do cliente
    avaliacao = None
    if status == 'Entregue':
        # Avaliação tende a ser menor se houver atraso
        if atraso > 0:
            avaliacao = random.randint(1, 3)
        else:
            avaliacao = random.randint(3, 5)
    
    pedido = {
        'pedido_id': i + 1,
        'cliente_id': cliente_id,
        'data_pedido': data_pedido,
        'valor_total': round(valor_total, 2),
        'metodo_pagamento': random.choice(metodo_pagamento),
        'status': status,
        'n_itens': n_itens
    }
    
    entrega = {
        'entrega_id': i + 1,
        'pedido_id': i + 1,
        'transportadora_id': transportadora_id,
        'data_envio': data_envio,
        'data_entrega_prevista': data_entrega_prevista,
        'data_entrega_real': data_entrega_real if status == 'Entregue' else None,
        'status': status,
        'atraso_dias': atraso if status == 'Entregue' else None,
        'avaliacao_cliente': avaliacao
    }
    
    pedidos.append(pedido)
    entregas.append(entrega)

df_pedidos = pd.DataFrame(pedidos)
df_entregas = pd.DataFrame(entregas)

# Mostrando estatísticas básicas dos dados gerados
print(f"\nClientes: {len(df_clientes)} registros")
print(f"Produtos: {len(df_produtos)} registros")
print(f"Transportadoras: {len(df_transportadoras)} registros")
print(f"Regiões: {len(df_regioes)} registros")
print(f"Pedidos: {len(df_pedidos)} registros")
print(f"Entregas: {len(df_entregas)} registros")

# Salvando os dados em CSV
df_clientes.to_csv('dados_brutos_clientes.csv', index=False)
df_produtos.to_csv('dados_brutos_produtos.csv', index=False)
df_transportadoras.to_csv('dados_brutos_transportadoras.csv', index=False)
df_regioes.to_csv('dados_brutos_regioes.csv', index=False)
df_pedidos.to_csv('dados_brutos_pedidos.csv', index=False)
df_entregas.to_csv('dados_brutos_entregas.csv', index=False)

# ----------------------------------------------------------------------
# ENTENDIMENTO DOS DADOS (DATA UNDERSTANDING)
# ----------------------------------------------------------------------
print("\n\n2. ENTENDIMENTO DOS DADOS (CRISP-DM)")
print("-" * 50)

# Examinando distribuição de atrasos
df_entregas_concluidas = df_entregas[df_entregas['status'] == 'Entregue'].copy()
df_entregas_concluidas['atrasada'] = df_entregas_concluidas['atraso_dias'] > 0

print("\nDistribuição de entregas atrasadas:")
print(df_entregas_concluidas['atrasada'].value_counts())
taxa_atraso = df_entregas_concluidas['atrasada'].mean() * 100
print(f"Taxa de atraso: {taxa_atraso:.2f}%")

# Estatísticas de atraso por dia
print("\nEstatísticas de atraso em dias:")
print(df_entregas_concluidas['atraso_dias'].describe())

# Relação entre atraso e avaliação do cliente
print("\nAvaliação média do cliente por status de atraso:")
print(df_entregas_concluidas.groupby('atrasada')['avaliacao_cliente'].mean())

# Analisando atrasos por transportadora
atrasos_por_transportadora = df_entregas_concluidas.groupby('transportadora_id')['atrasada'].mean().sort_values(ascending=False)
print("\nTaxa de atraso por transportadora:")
print(atrasos_por_transportadora)

# Analisando atrasos por região do cliente
df_entregas_com_regiao = pd.merge(
    df_entregas_concluidas, 
    pd.merge(df_pedidos, df_clientes[['cliente_id', 'regiao_id']], on='cliente_id'), 
    on='pedido_id'
)
atrasos_por_regiao = df_entregas_com_regiao.groupby('regiao_id')['atrasada'].mean().sort_values(ascending=False)
print("\nTaxa de atraso por região:")
print(atrasos_por_regiao)

# Analisando relação entre valor do pedido e atraso
print("\nValor médio de pedidos atrasados vs. não atrasados:")
print(df_entregas_com_regiao.groupby('atrasada')['valor_total'].mean())

# Analisando relação entre número de itens e atraso
print("\nNúmero médio de itens em pedidos atrasados vs. não atrasados:")
print(df_entregas_com_regiao.groupby('atrasada')['n_itens'].mean())

# Visualização de dados
plt.figure(figsize=(10, 6))
sns.countplot(x='atrasada', data=df_entregas_concluidas)
plt.title('Distribuição de Entregas Atrasadas vs. No Prazo')
plt.xticks([0, 1], ['No Prazo', 'Atrasada'])
plt.xlabel('Status da Entrega')
plt.ylabel('Número de Entregas')
plt.savefig('imagens\\distribuicao_atrasos.png')

# Relação entre atraso e avaliação
plt.figure(figsize=(10, 6))
sns.boxplot(x='atrasada', y='avaliacao_cliente', data=df_entregas_concluidas)
plt.title('Avaliação do Cliente por Status de Atraso')
plt.xticks([0, 1], ['No Prazo', 'Atrasada'])
plt.xlabel('Status da Entrega')
plt.ylabel('Avaliação do Cliente (1-5)')
plt.savefig('imagens\\avaliacao_vs_atraso.png')

# ----------------------------------------------------------------------
# PREPARAÇÃO DOS DADOS (DATA PREPARATION)
# ----------------------------------------------------------------------
print("\n\n3. PREPARAÇÃO DOS DADOS (CRISP-DM)")
print("-" * 50)

# 1. Combinando dados para criar um conjunto de dados unificado para análise
print("\n1. Criando conjunto de dados unificado para análise")

# Unindo dados de pedidos, clientes, entregas, transportadoras e regiões
df_analise = pd.merge(df_pedidos, df_entregas[
    ['pedido_id', 'transportadora_id', 'data_envio', 'data_entrega_prevista',
     'data_entrega_real', 'atraso_dias', 'avaliacao_cliente', 'status']
], on='pedido_id')

# Imprimir as colunas para verificar
print("Colunas após o primeiro merge:")
print(df_analise.columns.tolist())

df_analise = pd.merge(df_analise, df_clientes[
    ['cliente_id', 'regiao_id', 'cidade', 'estado']
], on='cliente_id')

df_analise = pd.merge(df_analise, df_transportadoras[
    ['transportadora_id', 'nome', 'tempo_medio_entrega_dias', 'taxa_atraso']
], on='transportadora_id')

df_analise = pd.merge(df_analise, df_regioes[
    ['regiao_id', 'nome', 'distancia_media_km', 'taxa_congestionamento']
], on='regiao_id')

print(f"Dimensões do conjunto de dados unificado: {df_analise.shape}")

# 2. Tratamento de datas
print("\n2. Tratamento de datas e criação de recursos temporais")

# Convertendo colunas de data para datetime
colunas_data = ['data_pedido', 'data_envio', 'data_entrega_prevista', 'data_entrega_real']
for col in colunas_data:
    if col in df_analise.columns:
        df_analise[col] = pd.to_datetime(df_analise[col])

# Criando features temporais
df_analise['tempo_processamento'] = (df_analise['data_envio'] - df_analise['data_pedido']).dt.days
df_analise['tempo_entrega_estimado'] = (df_analise['data_entrega_prevista'] - df_analise['data_envio']).dt.days

# Criando features de sazonalidade
df_analise['mes_pedido'] = df_analise['data_pedido'].dt.month
df_analise['dia_semana_pedido'] = df_analise['data_pedido'].dt.dayofweek
df_analise['fim_de_semana'] = df_analise['dia_semana_pedido'].isin([5, 6]).astype(int)

# 3. Tratamento de valores ausentes
print("\n3. Tratamento de valores ausentes")

valores_ausentes = df_analise.isnull().sum()
print("Valores ausentes por coluna:")
print(valores_ausentes[valores_ausentes > 0])

# Tratamento de entregas ainda não concluídas (data_entrega_real e atraso_dias são None)
# Para fins de análise, vamos filtrar apenas entregas concluídas
df_analise_concluidas = df_analise[df_analise['status_y'] == 'Entregue'].copy()

# 4. Engenharia de recursos (Feature Engineering)
print("\n4. Engenharia de recursos")

# Criando variável alvo para problema de classificação
df_analise_concluidas['atrasada'] = (df_analise_concluidas['atraso_dias'] > 0).astype(int)

# Criando indicadores de desempenho
df_analise_concluidas['eficiencia_entrega'] = df_analise_concluidas['tempo_entrega_estimado'] / \
                                             (df_analise_concluidas['data_entrega_real'] - df_analise_concluidas['data_envio']).dt.days

# Taxa de entrega no prazo por transportadora
taxa_sucesso_transportadora = df_analise_concluidas.groupby('transportadora_id')['atrasada'].mean()
df_transportadoras_desempenho = pd.DataFrame({'transportadora_id': taxa_sucesso_transportadora.index,
                                            'taxa_atraso_historica': taxa_sucesso_transportadora.values})

# Agregando dados históricos ao nível do cliente
df_historico_cliente = df_analise_concluidas.groupby('cliente_id').agg({
    'pedido_id': 'count',
    'valor_total': 'mean',
    'atrasada': 'mean',
    'avaliacao_cliente': 'mean'
}).reset_index()

df_historico_cliente.columns = ['cliente_id', 'n_pedidos', 'valor_medio', 'taxa_atraso', 'avaliacao_media']

# 5. Normalização e padronização
print("\n5. Normalização e padronização")

# Selecionando colunas numéricas para normalização
colunas_numericas = ['valor_total', 'tempo_processamento', 'tempo_entrega_estimado', 
                    'distancia_media_km', 'taxa_congestionamento']

# Aplicando padronização (z-score)
from sklearn.preprocessing import StandardScaler

# Aplicando o scaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_analise_concluidas[colunas_numericas])

# Criando um DataFrame com os valores escalados
scaled_df = pd.DataFrame(
    scaled_features,
    columns=[f"{col}_scaled" for col in colunas_numericas],
    index=df_analise_concluidas.index
)

# Concatenando com o DataFrame original
df_analise_concluidas = pd.concat([df_analise_concluidas, scaled_df], axis=1)

# 6. Codificação de variáveis categóricas
print("\n6. Codificação de variáveis categóricas")

# One-hot encoding para variáveis categóricas
df_analise_preparado = pd.get_dummies(
    df_analise_concluidas, 
    columns=['metodo_pagamento', 'estado'], 
    drop_first=True
)

# 7. Seleção de atributos para diferentes tipos de modelagem
print("\n7. Seleção de atributos para diferentes tipos de modelagem")

# Features para previsão de atrasos (Classificação)
features_classificacao = [col for col in df_analise_preparado.columns 
                         if col not in ['pedido_id', 'cliente_id', 'data_pedido', 'data_envio', 
                                        'data_entrega_prevista', 'data_entrega_real', 'status_x', 'status_y',
                                        'atraso_dias', 'avaliacao_cliente', 'atrasada']]
target_classificacao = 'atrasada'

print(f"\nNúmero de features para classificação: {len(features_classificacao)}")

# Features para previsão de tempo de entrega (Regressão)
features_regressao = [col for col in df_analise_preparado.columns 
                      if col not in ['pedido_id', 'cliente_id', 'data_pedido', 'data_envio', 
                                    'data_entrega_prevista', 'data_entrega_real', 'status_x', 'status_y',
                                    'atraso_dias', 'avaliacao_cliente', 'atrasada']]
target_regressao = 'atraso_dias'

print(f"Número de features para regressão: {len(features_regressao)}")

# 8. Divisão em conjuntos de treino e teste
print("\n8. Divisão em conjuntos de treino e teste")

from sklearn.model_selection import train_test_split

X = df_analise_preparado[features_classificacao]
y = df_analise_preparado[target_classificacao]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dimensões do conjunto de treino: {X_train.shape}")
print(f"Dimensões do conjunto de teste: {X_test.shape}")

# Salvando os dados processados
df_analise_preparado.to_csv('dados_preparados.csv', index=False)

print("\n9. Dados preparados e salvos para a fase de modelagem")
print("O processo de entendimento e preparação de dados está concluído.")