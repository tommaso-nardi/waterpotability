import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Carica il dataset
dataset = pd.read_csv("water_potability_corrected.csv")

# Di questi sceglierne solo uno! Il primo cancella le entry con dati mancanti, la seconda invece fa l'imputazione con media
# dataset = dataset.dropna()
dataset.fillna(dataset.mean(), inplace=True)

# Separazione di features X (tutti i valori delle feature tranne Potability)
# dal target y (cioè la "Potability", il valore da predire)
X = dataset.drop("Potability", axis=1)
y = dataset["Potability"]

# Suddivisione del dataset in training e testing, SMOTE ne ha bisogno
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utilizzo di SMOTE per i campioni sintetici
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Copia del dataset originale
balanced_dataset = dataset.copy()

# Salvataggio del nuovo dataset, X_resampled ha i valori, y_resampled ha la Potabilità
balanced_dataset = pd.DataFrame(X_resampled, columns=X.columns)
balanced_dataset['Potability'] = y_resampled
balanced_dataset.to_csv("water_potability_corrected_extended.csv", index=False)