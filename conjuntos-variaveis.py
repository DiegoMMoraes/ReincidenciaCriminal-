import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools

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
races = ['race_African-American','race_Caucasian','race_Hispanic']

# Armazenar os resultados
results = {}

for var in races:
    # Dividir os dados com base na variável
    df_group11 = df[(df[var] == 1) & (df['sex_Male'] == 1)]  
    df_group10 = df[(df[var] == 1) & (df['sex_Male'] == 0)]  
    df_group01 = df[(df[var] == 0) & (df['sex_Male'] == 1)]  
    df_group00 = df[(df[var] == 0) & (df['sex_Male'] == 0)] 
    
    # Ajustar e avaliar o modelo para os grupos, se não estiverem vazios
    if not df_group11.empty:
        accuracy_group11, weights_group11 = ajustar_modelo(df_group11)
    else:
        accuracy_group11, weights_group11 = None, None

    if not df_group10.empty:
        accuracy_group10, weights_group10 = ajustar_modelo(df_group10)
    else:
        accuracy_group10, weights_group10 = None, None

    if not df_group01.empty:
        accuracy_group01, weights_group01 = ajustar_modelo(df_group01)
    else:
        accuracy_group01, weights_group01 = None, None

    if not df_group00.empty:
        accuracy_group00, weights_group00 = ajustar_modelo(df_group00)
    else:
        accuracy_group00, weights_group00 = None, None
        
    # Armazenar os resultados
    results[var] = {
        'accuracy_group11': accuracy_group11,
        'weights_group11': weights_group11,

        'accuracy_group10': accuracy_group10,
        'weights_group10': weights_group10,

        'accuracy_group01': accuracy_group01,
        'weights_group01': weights_group01,

        'accuracy_group00': accuracy_group00,
        'weights_group00': weights_group00,
    }

# Imprimir os resultados
for var, result in results.items():
    print(f"\nVariável: {var}")
    
    if result['accuracy_group11'] is not None:
        print("Taxa de Acerto para Grupo 11 =", result['accuracy_group11'])
        print("Pesos para cada variavel independente para Grupo 11")
        print(result['weights_group11'])
    else:
        print("Grupo 11 está vazio.")
        
    if result['accuracy_group10'] is not None:
        print("\nTaxa de Acerto para Grupo 10 =", result['accuracy_group10'])
        print("Pesos para cada variavel independente para Grupo 10")
        print(result['weights_group10'])
    else:
        print("Grupo 10 está vazio.")
        
    if result['accuracy_group01'] is not None:
        print("\nTaxa de Acerto para Grupo 01 =", result['accuracy_group01'])
        print("Pesos para cada variavel independente para Grupo 01")
        print(result['weights_group01'])
    else:
        print("Grupo 01 está vazio.")
        
    if result['accuracy_group00'] is not None:
        print("\nTaxa de Acerto para Grupo 00 =", result['accuracy_group00'])
        print("Pesos para cada variavel independente para Grupo 00")
        print(result['weights_group00'])
    else:
        print("Grupo 00 está vazio.")

# Visualização dos resultados
# for var, result in results.items():
#     if result['weights_group11'] is not None and result['weights_group10'] is not None and result['weights_group01'] is not None and result['weights_group00'] is not None:
#         plt.figure(figsize=(12, 6))
#         result['weights_group11'].plot(kind='bar', alpha=0.5, label='Grupo 11', color='blue')
#         result['weights_group10'].plot(kind='bar', alpha=0.5, label='Grupo 10', color='green')
#         result['weights_group01'].plot(kind='bar', alpha=0.5, label='Grupo 01', color='red')
#         result['weights_group00'].plot(kind='bar', alpha=0.5, label='Grupo 00', color='purple')
#         plt.title(f'Pesos das Variáveis Independentes para {var}')
#         plt.xlabel('Variáveis Independentes')
#         plt.ylabel('Pesos')
#         plt.legend()
#         plt.show()
