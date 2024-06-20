import pandas as pd

# Dados fornecidos
data = {
    'Características': [
        'sex_Male e race_African-american', 'sex_Female e race_African-american',
        'sex_Male e não race_African-american', 'sex_Female e não race_African-american',
        'sex_male e race_Caucasian', 'sex_Female e race_Caucasian',
        'sex_Male e não race_Caucasian', 'sex_Female e não race_Caucasian',
        'sex_male e race_Hispanic', 'sex_Female e race_Hispanic',
        'sex_Male e não race_Hispanic', 'sex_Female e não race_Hispanic'
    ],
    'Taxa de Acerto da Regressão Usada': [
        0.697, 0.724, 0.697, 0.757, 0.697, 0.735, 0.697, 0.718, 0.697, 0.8571, 0.697, 0.718
    ],
    'Quantidade de Observações': [
        2626, 549, 2371, 626, 1621, 482, 3376, 693, 427, 82, 4570, 1093
    ]
}

# Criar DataFrame
df = pd.DataFrame(data)

# Calcular a taxa de acurácia média ponderada
taxa_acuracia_media_ponderada = (df['Taxa de Acerto da Regressão Usada'] * df['Quantidade de Observações']).sum() / (df['Quantidade de Observações']).sum()

print("Taxa de Acurácia Média Ponderada:", taxa_acuracia_media_ponderada)
