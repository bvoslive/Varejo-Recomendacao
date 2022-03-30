# IMPORTANDO BIBLIOTECAS
import pandas as pd
import xlrd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import KernelPCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score


import re, seaborn as sns, numpy as np, pandas as pd, random
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


df_2000.columns


x = df_2000.iloc[:, 2:]


# NORMALIZAÇÃO

x_normalizado = StandardScaler().fit_transform(x)

# KERNEL PCA

transformer = KernelPCA(n_components=4, kernel='rbf', random_state=42)
transformer.fit(x_normalizado)
x_transformado = transformer.transform(x_normalizado)


print(transformer.eigenvalues_)

plt.scatter(x_transformado[:, 0], x_transformado[:, 1], alpha=0.2)
plt.show()


# AFFINITY PROPAGATION

clustering = AffinityPropagation(random_state=42, damping=0.6, affinity='precomputed')


clustering.fit(x_transformado)
labels = clustering.labels_

pd.Series(labels).value_counts()



#MÉTRICA

silhouette_score(x_transformado, labels)
# SEM KERNEL PCA = 20%



# VENDO SE FAZ SENTIDO

df_2000[df_2000['Município'].str.contains('São Paulo')]

# SÃO PAULO
pd.Series(labels)[4818]

labels_serie = pd.Series(labels)

indices_pesquisar = labels_serie[labels_serie == 33].index.tolist()


df_2000.columns

teste_hist = df_2000.iloc[indices_pesquisar]['Intensidade da pobreza, 2000']

sns.distplot(teste_hist)
plt.show()



