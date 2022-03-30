# IMPORTANDO BIBLIOTECAS
import pandas as pd
import xlrd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import KernelPCA
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score


import seaborn as sns
from pylab import *
from matplotlib.pyplot import plot, show, draw, figure, cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# IMPORTANDO DADOS
df = pd.read_excel('Case Seleção 2022 -DATA SCIENTIST.xls', sheet_name='Informações_Municipais', header=3)

# DATA CLEANING
df['População urbana, 2000'].describe()

# Eliminando 1991
df_2000 = df.drop(['População de 25 anos ou mais de idade, 1991', 'População de 65 anos ou mais de idade, 1991', 'População total, 1991'], axis=1)

# Evitando colinearidade
df_2000.drop(['População total, 2000', 'Área (km²)'], axis=1, inplace=True)
x = df_2000.iloc[:, 2:]

# NORMALIZAÇÃO
scaler = StandardScaler()
x_normalizado = scaler.fit_transform(x)





# KERNEL PCA
transformer = KernelPCA(n_components=4, kernel='rbf', random_state=42)
transformer.fit(x_normalizado)
x_transformado = transformer.transform(x_normalizado)









# IMPRIMINDO AUTOVALORES
print(transformer.eigenvalues_)

# VISUALIZANDO DADOS
plt.scatter(x_transformado[:, 0], x_transformado[:, 1], alpha=0.2)
plt.show()

# AFFINITY PROPAGATION
clustering = AffinityPropagation(random_state=42, damping=0.6, max_iter=300, convergence_iter=40)
clustering.fit(x_transformado)

labels = clustering.labels_

pd.Series(labels).value_counts()


# EXECUTANDO PREDIÇÃO

cidade = 'São Paulo'

municipio = df_2000[df_2000['Município'].str.contains(cidade)].iloc[0]
municipio_scaled = scaler.transform([municipio.iloc[2:]])
municipio_transformed = transformer.transform(municipio_scaled)
label_predicted = clustering.predict(municipio_transformed)
label_predicted = label_predicted.tolist()
label_predicted = label_predicted[0]


# FILTRANDO ESCOLHAS
labels_serie = pd.Series(labels)
indices_pesquisar = labels_serie[labels_serie == label_predicted].index.tolist()

matriz_afinidade = pd.DataFrame(clustering.affinity_matrix_)
pesquisa_afinidade = matriz_afinidade.loc[indices_pesquisar, indices_pesquisar]

eixos = df_2000['Município'][indices_pesquisar]
eixos = eixos.tolist()

pesquisa_afinidade.columns = eixos
pesquisa_afinidade.index = eixos





# VISUALIZANDO POSSÍVEIS ESCOLHAS
sns.heatmap(pesquisa_afinidade)
plt.show()

THRESHOLD = -0.01
pesquisa_afinidade_media = pesquisa_afinidade.median(axis=0)
pesquisa_afinidade_media = pesquisa_afinidade_media[pesquisa_afinidade_media > THRESHOLD]

recomendacoes = pesquisa_afinidade_media.sort_values(ascending=False)
recomendacoes = recomendacoes.index.tolist()


pd.DataFrame(recomendacoes).to_excel('Cidades recomendadas.xlsx')


#----------------------------------------

# INSIGHTS
pd.Series(labels)[606]

labels_serie = pd.Series(labels)
indices_pesquisar = labels_serie[labels_serie == 33].index.tolist()

df_selecao = df_2000.loc[indices_pesquisar]

df_selecao.columns

sns.distplot(df_selecao['Taxa de alfabetização, 2000'])
plt.show()








