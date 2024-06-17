import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import itertools

# Carrega o arquivo
df = pd.read_csv("ProPublica.csv")

# Excluindo colunas inuteis
df.drop(['Unnamed: 0', 'sex_Female', 'degree_F'], axis=1, inplace=True)

# Verificar colunas presentes no DataFrame
print("Colunas disponíveis no DataFrame:")
print(df.columns)

# Função para ajustar e avaliar o modelo
def ajustar_modelo(df, features):
    Y = df["two_year_recid"]
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=20)
    model = LogisticRegression(max_iter=6000)
    model.fit(X_train, y_train)
    prediction_test = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction_test)
    return accuracy

# Variáveis independentes para testar (sexo e raça)
# Ajustar com base nas colunas realmente existentes no DataFrame
sexo_raca_vars = ['sex_Male', 'race_African-American', 'race_Caucasian', 'race_Hispanic', 'race_Other']

# Armazenar os resultados
results = {}

# Gerar todas as combinações de até três características de sexo e raça
for i in range(1, 3):
    for combo in itertools.combinations(sexo_raca_vars, i):
        combo_name = '_'.join(combo)
        
        # Verificar se todas as colunas da combinação existem no DataFrame
        if all(col in df.columns for col in combo):
            # Dividir os dados com base na combinação de características
            df_group1 = df[df[list(combo)].sum(axis=1) > 0]
            df_group2 = df[df[list(combo)].sum(axis=1) == 0]
            
            if len(df_group1) > 0 and len(df_group2) > 0:
                # Ajustar e avaliar o modelo para o grupo 1 (onde a combinação de características é 1)
                accuracy_group1 = ajustar_modelo(df_group1, list(combo))
                
                # Ajustar e avaliar o modelo para o grupo 2 (onde a combinação de características é 0)
                accuracy_group2 = ajustar_modelo(df_group2, list(combo))
                
                # Armazenar os resultados
                results[combo_name] = {
                    'accuracy_group1': accuracy_group1,
                    'accuracy_group2': accuracy_group2
                }

# Imprimir os resultados
for combo, result in results.items():
    print(f"\nCombinação: {combo}")
    print("Taxa de Acerto para Grupo 1 =", result['accuracy_group1'])
    print("Taxa de Acerto para Grupo 2 =", result['accuracy_group2'])
