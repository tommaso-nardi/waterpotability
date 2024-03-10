import pandas as pd

# Carica il dataset e imputalo con media per evitare che alcune tuple si "salvino"
dataset = pd.read_csv("water_potability.csv")
dataset.fillna(dataset.mean(), inplace=True)

# Lista di valori ph specifici da correggere
ph_errore = [1,2,3,4,5,10,11,12,13,14]

# Individua le entry che hanno valori "impossibili"
error_ph = dataset[(dataset["ph"].isin(ph_errore)) & (dataset["Potability"] == 1)].index
error_hardness = dataset[(dataset["Hardness"] > 230) & (dataset["Potability"] == 1)].index
error_carbon = dataset[(dataset["Organic_carbon"] > 13) & (dataset["Potability"] == 1)].index
error_trihalomethanes = dataset[(dataset["Trihalomethanes"] > 100) & (dataset["Potability"] == 1)].index
error_turbidity = dataset[(dataset["Turbidity"] > 5) & (dataset["Potability"] == 1)].index

# Correggi gli errori impostando la classe a "Non Potabile" per gli indici identificati
dataset.loc[error_ph, "Potability"] = 0
dataset.loc[error_hardness, "Potability"] = 0
dataset.loc[error_carbon, "Potability"] = 0
dataset.loc[error_trihalomethanes, "Potability"] = 0
dataset.loc[error_turbidity, "Potability"] = 0

# Salva il dataset corretto
dataset.to_csv("water_potability_corrected.csv", index=False)
