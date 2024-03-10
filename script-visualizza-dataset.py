import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Carica il dataset
dataset = pd.read_csv("water_potability.csv")

# Creazione dell istogramma con conteggio delle potabilità
plt.figure(figsize=(10, 6))
sns.countplot(x='Potability', data=dataset, hue='Potability', palette={0: "#ff6b4a", 1: "#534aff"}, dodge=False)
plt.title('Distribuzione della Potabilità nel Dataset')
plt.xlabel('Potabilità')
plt.ylabel('Conteggio')
plt.show()

# Creazione del grafico a torta
plt.figure(figsize=(8, 8))
dataset['Potability'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
plt.title('Percentuale di Potabilità nel Dataset')
plt.show()
