import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
print("Correlation matrix:\n", correlation_matrix)

# Visualiza a matriz de correlação com um heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Excluindo colunas inuteis
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['sex_Female'], axis=1, inplace=True)
df.drop(['degree_F'], axis=1, inplace=True)
print(df.head(10))
