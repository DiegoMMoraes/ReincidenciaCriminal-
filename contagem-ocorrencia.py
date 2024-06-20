import pandas as pd

# Crie o DataFrame com base na tabela fornecida
df = pd.read_csv("ProPublica.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['sex_Female'], axis=1, inplace=True)
df.drop(['degree_F'], axis=1, inplace=True)


races = ['race_African-American','race_Caucasian','race_Hispanic']
for race in races:
    # Filtrar o DataFrame para a combinação específica de características
    filtered_df11 = df[(df['sex_Male'] == 1) & (df[race] == 1)]
    filtered_df10 = df[(df['sex_Male'] == 0) & (df[race] == 1)]
    filtered_df01 = df[(df['sex_Male'] == 1) & (df[race] == 0)]
    filtered_df00 = df[(df['sex_Male'] == 0) & (df[race] == 0)]
    print("Quantidade de observações com " + race + " e sex_Male: ", len(filtered_df11))
    print("Quantidade de observações com " + race + " e sex_Female: ", len(filtered_df10))
    print("Quantidade de observações com não " + race + " e sex_Male: ", len(filtered_df01))
    print("Quantidade de observações com não " + race + " e sex_Female: ", len(filtered_df00))