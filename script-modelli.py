import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Carica il dataset
dataset = pd.read_csv("water_potability.csv")
# Stampa dei valori nel dataset
dataset.info()

# Di questi sceglierne solo uno! Il primo cancella le entry con dati mancanti, la seconda invece fa l'imputazione con media
# dataset = dataset.dropna()
dataset.fillna(dataset.mean(), inplace=True)


# Separazione di features X (tutti i valori delle feature tranne Potability)
# dal target y (cioè la "Potability", il valore da predire)
X = dataset.drop("Potability", axis=1)
y = dataset["Potability"]

# Divisione del dataset in set di addestramento e test, 20% Test, 80% Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista dei modelli che useremo, nell'array "models"
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Per ogni modello che abbiamo definito...
for name, model in models.items():

    # Addestralo
    model.fit(X_train, y_train)

    # Effettua la predizione, y_pred è un array binario di tutte le previsioni, IN ORDINE
    y_pred = model.predict(X_test)

    # Calcolo delle varie metriche confrontando il valore reale con quello predetto (0 o 1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    #Stampa di tutte le metriche studiate nel corso, con fino a 9 cifre dopo la virgola
    print(f"\nMetriche di{name}:")
    print(f"Accuracy: {accuracy:.9f}")
    print(f"Precision: {precision:.9f}")
    print(f"Recall: {recall:.9f}")
    print(f"F1-Score: {f1:.9f}")
    print("Confusion Matrix:")
    print(cm)

    # Creazione Immagine della Curva di Apprendimento (come si comporta il modello al variare della dimensione
    # del dataset nel corso del tempo usando la cross-validation delle 5 train_sizes
    # ricordare che questa curva fa riferimento all'accuracy, come specificato in "scoring")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0]
    )
    train_mean = train_scores.mean(axis=1)  #Media e Deviazione su dati training, più alta la deviazione, più vari i dati
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)    #Media e Deviazione su dati di test, più alta la deviazione, più vari i dati
    test_std = test_scores.std(axis=1)

    plt.figure()
    # La Training Accuracy va ad indicare quanto il modello impara bene dai dati di training
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    # La Testing Accuracy altro non è che il variare dell'Accuracy man mano che aggiungiamo dati di test
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Testing Accuracy')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
    plt.xlabel('Numero di dati Training')
    plt.ylabel('Accuracy')
    plt.title(f'Curva Apprendimento - {name}')
    plt.legend(loc='lower right')
    plt.show()

    # Creazione Immagine della Curva ROC
    if hasattr(model, "predict_proba"):
        # probas ha il numero di casi in cui è stato predetto 1
        probas = model.predict_proba(X_test)[:, 1]
        # Calcolo della curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, probas)
        # Calcolo dell'area sotto la curva
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (Area Sotto La Curva = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Falsi Positivi')
        plt.ylabel('Veri Positivi')
        plt.title(f'ROC - {name}')
        plt.legend(loc='lower right')
        plt.show()