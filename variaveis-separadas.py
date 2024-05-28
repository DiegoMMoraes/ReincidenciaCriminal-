import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Carrega o arquivo
df = pd.read_csv("ProPublica.csv")

# Excluindo colunas inuteis
df.drop(['Unnamed: 0', 'sex_Female', 'degree_F'], axis=1, inplace=True)

# Função para ajustar e avaliar o modelo
def ajustar_modelo(df):
    Y = df["two_year_recid"]
    X = df.drop('two_year_recid', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=20)
    model = LogisticRegression(max_iter=6000)
    model.fit(X_train, y_train)
    prediction_test = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction_test)
    weights = pd.Series(model.coef_[0], index=X.columns.values)
    return accuracy, weights

# Variáveis independentes para testar
independent_variables = df.drop('two_year_recid', axis=1).columns

# Armazenar os resultados
results = {}

for var in independent_variables:
    # Dividir os dados com base na variável
    df_group1 = df[df[var] == 1]
    df_group2 = df[df[var] == 0]
    
    if len(df_group1) > 0 and len(df_group2) > 0:
        # Ajustar e avaliar o modelo para o grupo 1 (onde a variável é 1)
        accuracy_group1, weights_group1 = ajustar_modelo(df_group1)
        
        # Ajustar e avaliar o modelo para o grupo 2 (onde a variável é 0)
        accuracy_group2, weights_group2 = ajustar_modelo(df_group2)
        
        # Armazenar os resultados
        results[var] = {
            'accuracy_group1': accuracy_group1,
            'weights_group1': weights_group1,
            'accuracy_group2': accuracy_group2,
            'weights_group2': weights_group2
        }

# Imprimir os resultados
for var, result in results.items():
    print(f"\nVariável: {var}")
    print("Taxa de Acerto para Grupo 1 =", result['accuracy_group1'])
    print("Pesos para cada variavel independente para Grupo 1")
    print(result['weights_group1'])
    print("\nTaxa de Acerto para Grupo 2 =", result['accuracy_group2'])
    print("Pesos para cada variavel independente para Grupo 2")
    print(result['weights_group2'])

# Visualização dos resultados
for var, result in results.items():
    plt.figure(figsize=(12, 6))
    result['weights_group1'].plot(kind='bar', alpha=0.5, label='Grupo 1', color='blue')
    result['weights_group2'].plot(kind='bar', alpha=0.5, label='Grupo 2', color='red')
    plt.title(f'Pesos das Variáveis Independentes para {var}')
    plt.xlabel('Variáveis Independentes')
    plt.ylabel('Pesos')
    plt.legend()
    plt.show()


# O QUE FALTA:
    # pq o age não está sendo calculado