import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
dataset = pd.read_csv("CSV Files/water_potability_corrected_extended.csv")

# Di questi sceglierne solo uno! Il primo cancella le entry con dati mancanti, la seconda invece fa l'imputazione con media
# dataset = dataset.dropna()
dataset.fillna(dataset.mean(), inplace=True)

# Separazione delle entry potabili e non potabili
non_potabile = dataset.query("Potability == 0")
potabile = dataset.query("Potability == 1")

# Creazione del plot che conterrà tutte le distribuzioni
plt.figure(figsize=(15, 15))
for ax, col in enumerate(dataset.columns[:-1]):
    plt.subplot(3, 3, ax + 1)
    plt.title(col)
    plotting = sns.kdeplot(x=non_potabile[col], label="Non Potabile", fill=True, common_norm=False, color="#ff6b4a", alpha=0.9, linewidth=3)
    plotting = sns.kdeplot(x=potabile[col], label="Potabile", fill=True, common_norm=False, color="#534aff", alpha=0.9, linewidth=3)
    plt.legend()

plt.tight_layout()
plt.suptitle('Distribuzione delle Features', y=1.08, size=26, weight='bold')
plt.show()
