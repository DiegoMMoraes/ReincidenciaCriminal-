import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

# Carrega o arquivo
df = pd.read_csv("ProPublica.csv")

# Desenha grafos de dispersão para duas variáveis quaisquer selecionadas
# plt.scatter(df['sex_Female'], df['two_year_recid'], marker='+', color='red')
# plt.xlabel('sex_Female')
# plt.ylabel('two_year_recid')
# plt.title('Grafico de dispersão para "sex female" e "two year recid" ')
# plt.show()

# plt.scatter(df['age'], df['two_year_recid'], marker='+', color='blue')
# plt.xlabel('age')
# plt.ylabel('two_year_recid')
# plt.title('Grafico de dispersão para "age" e "two year recid" ')
# plt.show()

# plt.scatter(df['race_African-American'], df['two_year_recid'], marker='+', color='blue')
# plt.xlabel('age')
# plt.ylabel('two_year_recid')
# plt.title('Grafico de dispersão para "race_African-American" e "two year recid" ')
# plt.show()

# Calcula a matriz de correlação
correlation_matrix = df.corr()

# Visualiza a matriz de correlação com um heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix Heatmap')
# plt.show()

# Excluindo colunas inuteis
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['sex_Female'], axis=1, inplace=True)
df.drop(['degree_F'], axis=1, inplace=True)

# separa as variaveis independentes e dependentes
Y = df["two_year_recid"]
X = df.drop('two_year_recid', axis=1)

# Separa as observaçoes de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=20)

# print(X.head())
# print(Y.head())

# print(X_train.head())
# print(y_train.head())

# Cria o modelo de regressao logistica
regressaoGeral = LogisticRegression(max_iter=6000)
regressaoGeral.fit(X_train, y_train)

# Desobre a taxa de acerto
prediction_test = regressaoGeral.predict(X_test)
print ("Taxa de Acerto = ", metrics.accuracy_score(y_test, prediction_test))

weights = pd.Series(regressaoGeral.coef_[0], index=X.columns.values)

print("Pesos para cada variavel independente")
print(weights)