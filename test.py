import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carregar os dados em um DataFrame
data = pd.read_csv('ProPublica.csv')  # Supondo que você converteu o CSV da imagem para um arquivo

# Verificar os dados
print(data.head(6174))

# Dividir os dados em variáveis independentes (X) e variável dependente (y)
X = data.drop('two_year_recid', axis=1)
y = data['two_year_recid']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Construir o modelo de Regressão Logística
model = LogisticRegression(max_iter=5000)  # max_iter aumentado para garantir a convergência
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))