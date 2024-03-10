import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Carica il dataset
dataset = pd.read_csv("water_potability_corrected_extended.csv")
#Stampa delle linee guida su console per specificare all'utente che dati indicano cosa
print("Linee guida:\nph: 0-14\nHardness: Calcio e Magnesio nell'acqua in mg/L\nSolids: Solidi Disciolti nell'acqua in ppm\nChloramines: Cloramina (un agente disinfattante) nell'acqua in ppm\nSulfate: Solfato nell'acqua in mg/L\nConductivity: Capacità di condurre elettricità nell'acqua in microsiemens per centimetro (μS/cm)\nOrganic_Carbon: Il carbonio nell'acqua in ppm\nTrihalomethanes: Trialometani (sottoprodotti della disinfezione) nell'acqua in microgrammi per litro (μg/L)\nTurbidity: Turbidità (la capacità dell'acqua di diffondere la luce) in Unità Nefelometriche di Turbidità (NTU)\n")

# Imputazione tramite Media (mean) in caso ci dovessero essere problemi in lettura
dataset.fillna(dataset.mean(), inplace=True)

# Separazione delle features X (tutti i valori delle feature tranne Potability)
# dal target y (cioè la "Potability", il valore da predire)
X = dataset.drop("Potability", axis=1)
y = dataset["Potability"]

# Creiamoci il RandomForest...
randomforest = RandomForestClassifier(random_state=42)
randomforest.fit(X, y)

# E qui la variabile per continuare
continua = True

while continua:
    # Chiedi all'utente di inserire i valori per una nuova entry (tranne "Potability")
    new_entry = {}
    for column in X.columns:
        value = input(f"Inserisci il valore per '{column}': ")
        new_entry[column] = float(value)

    # Creazione di un DataFrame con la nuova entry
    new_entry_df = pd.DataFrame([new_entry])

    # Effettua la predizione sul nuovo dato
    prediction = randomforest.predict(new_entry_df)

    # Stampa il risultato
    if prediction[0] != 1:
        print("L'acqua non è potabile.")
    else:
        print("L'acqua è potabile.")

    ris = input(f"Vuoi inserire un'altra entry? (y/n): ")
    if ris.lower() != 'y':
        continua = False
