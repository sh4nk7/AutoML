#AutoML nel Settore Automobilistico Predizione del Comportamento di Acquisto nel Settore Automobilistico utilizzando il Machine Learning

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

#Importiamo il dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#conteggiamo i valori all'interno del nostro dataset
dataset.count()

# @title Purchased

from matplotlib import pyplot as plt
dataset['Purchased'].plot(kind='line', figsize=(8, 4), title='Purchased')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title EstimatedSalary

from matplotlib import pyplot as plt
dataset['EstimatedSalary'].plot(kind='line', figsize=(8, 4), title='EstimatedSalary')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title Age

from matplotlib import pyplot as plt
dataset['Age'].plot(kind='line', figsize=(8, 4), title='Age')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title User ID

from matplotlib import pyplot as plt
dataset['User ID'].plot(kind='line', figsize=(8, 4), title='User ID')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title EstimatedSalary vs Purchased

from matplotlib import pyplot as plt
dataset.plot(kind='scatter', x='EstimatedSalary', y='Purchased', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Age vs EstimatedSalary

from matplotlib import pyplot as plt
dataset.plot(kind='scatter', x='Age', y='EstimatedSalary', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title User ID vs Age

from matplotlib import pyplot as plt
dataset.plot(kind='scatter', x='User ID', y='Age', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Gender

from matplotlib import pyplot as plt
import seaborn as sns
dataset.groupby('Gender').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)


from matplotlib import pyplot as plt
import seaborn as sns

# Creazione del grafico a torta
plt.figure(figsize=(8, 6))  # Impostazione della dimensione della figura
dataset.groupby('Gender').size().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Dark2'), startangle=140)

# Rimozione delle spine superiori e destre del grafico
plt.gca().spines[['top', 'right']].set_visible(False)

# Aggiunta di una legenda
plt.legend(loc='upper right')

# Mostra il grafico
plt.show()

# @title Purchased

from matplotlib import pyplot as plt
dataset['Purchased'].plot(kind='hist', bins=20, title='Purchased')
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title EstimatedSalary

from matplotlib import pyplot as plt
dataset['EstimatedSalary'].plot(kind='hist', bins=20, title='EstimatedSalary')
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Age

from matplotlib import pyplot as plt
dataset['Age'].plot(kind='hist', bins=20, title='Age')
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title User ID

from matplotlib import pyplot as plt
dataset['User ID'].plot(kind='hist', bins=20, title='User ID')
plt.gca().spines[['top', 'right',]].set_visible(False)

#Primi dieci valori della lista

dataset.head(10)

#coda della lista

dataset.tail()

#tipo di dati

dataset.info()

#Descrizione

dataset.describe()

#Generiamo due dataset nei quali andiamo a inserire in uno solo le persone di sesso maschil (dataset_male) e nell'altro solo le persone di sesso femminile (dataset_female). Utilizzando una funzione di pandas.

dataset_male = dataset.loc[dataset['Gender']== 'Male']
dataset_male.count()
dataset_male.head()

dataset_female = dataset.loc[dataset['Gender']== 'Female']
dataset_female.count()
dataset_female.head()

# Assegnazione dei dati alle variabili X e y
X = dataset.iloc[:, [2, 3]].values  # Dati dalla seconda e terza colonna
y = dataset.iloc[:, 4].values  # Dati dalla quarta colonna (Purchased)

from sklearn.model_selection import train_test_split

# Divisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train e y_train saranno il training set
# X_test e y_test saranno il test set


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Creazione e addestramento del classificatore Perceptron
Perceptron_classification = Perceptron(max_iter=40, tol=0.01, eta0=0.1, random_state=0)
Perceptron_classification.fit(X_train, y_train)

model_classification = []  # Inizializza la lista vuota per memorizzare i modelli
score_classification = []  # Inizializza la lista vuota per memorizzare i punteggi di accuratezza
recall_classification = []  # Inizializza la lista vuota per memorizzare i punteggi di richiamo
precision_classification = []  # Inizializza la lista vuota per memorizzare i punteggi di precisione

# Valutazione delle prestazioni del classificatore sul test set
precision_Perceptron = precision_score(y_test, Perceptron_classification.predict(X_test))
score_Perceptron = accuracy_score(y_test, Perceptron_classification.predict(X_test))
recall_Perceptron = recall_score(y_test, Perceptron_classification.predict(X_test))
precision_Perceptron = precision_score(y_test, Perceptron_classification.predict(X_test), zero_division=1)

# Aggiunta delle metriche alla lista delle prestazioni del modello
model_classification.append('Perceptron')
score_classification.append(score_Perceptron)
recall_classification.append(recall_Perceptron)
precision_classification.append(precision_Perceptron)

# Stampare i valori ottenuti
print("model_classification:", model_classification)
print("score_classification:", score_classification)
print("recall_classification:", recall_classification)
print("precision_Perceptron",precision_Perceptron)

#Per avere una visione d'insieme per visualizzare graficamente le risposte utilizziamo:

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plt.title('Perceptron')
plot_decision_regions(X_test, y_test, clf=Perceptron_classification, legend=2)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Creazione e addestramento del modello di regressione logistica
Logistic_Regression_classification = LogisticRegression()
Logistic_Regression_classification.fit(X_train, y_train)

# Inizializzazione delle liste
model_Logistic_Regression = []
scores_Logistic_Regression = []
recalls_Logistic_Regression = []  # Lista per il richiamo
precisions_Logistic_Regression = []  # Lista per la precisione

# Calcolo delle metriche di valutazione
y_pred_Logistic_Regression = Logistic_Regression_classification.predict(X_test)
score_Logistic_Regression = accuracy_score(y_test, y_pred_Logistic_Regression)
recall_Logistic_Regression = recall_score(y_test, y_pred_Logistic_Regression)
precision_Logistic_Regression = precision_score(y_test, y_pred_Logistic_Regression, zero_division=0)

# Aggiunta del modello alla lista dei modelli
model_Logistic_Regression.append('Logistic_Regression')

# Aggiunta del punteggio di accuratezza alla lista dei punteggi
scores_Logistic_Regression.append(score_Logistic_Regression)

# Aggiunta del punteggio di richiamo alla lista dei punteggi
recalls_Logistic_Regression.append(recall_Logistic_Regression)

# Aggiunta del punteggio di precisione alla lista dei punteggi
precisions_Logistic_Regression.append(precision_Logistic_Regression)

# Stampa dei risultati
print("Modello:", model_Logistic_Regression[0])
print("Accuratezza:", scores_Logistic_Regression[0])
print("Recall:", recalls_Logistic_Regression[0])
print("Precisione:", precisions_Logistic_Regression[0])

# Visualizzazione delle regioni decisionali
plot_decision_regions(X_test, y_test, clf=Logistic_Regression_classification, legend=2)
plt.title('Logistic Regression')
plt.show()




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Creazione e addestramento del modello Random Forest
RandomForest_classification = RandomForestClassifier(random_state=5)
RandomForest_classification.fit(X_train, y_train)

# Inizializzazione delle liste se non sono già definite
model_classification = []
score_classification = []
recall_classification = []
precision_classification = []

# Calcolo delle metriche di valutazione
y_pred_RandomForest = RandomForest_classification.predict(X_test)
score_RandomForest = accuracy_score(y_test, y_pred_RandomForest)
recall_RandomForest = recall_score(y_test, y_pred_RandomForest)
precision_RandomForest = precision_score(y_test, y_pred_RandomForest, zero_division=0)

# Aggiunta del modello alla lista dei modelli
model_classification.append('Random Forest')

# Aggiunta del punteggio di accuratezza alla lista dei punteggi
score_classification.append(score_RandomForest)

# Aggiunta del punteggio di richiamo alla lista dei punteggi
recall_classification.append(recall_RandomForest)

# Aggiunta del punteggio di precisione alla lista dei punteggi
precision_classification.append(precision_RandomForest)

# Stampa dei risultati
print("Modello:", model_classification[0])
print("Accuratezza:", score_classification[0])
print("Recall:", recall_classification[0])
print("Precisione:", precision_classification[0])

# Visualizzazione delle regioni decisionali
plot_decision_regions(X_test, y_test, clf=RandomForest_classification, legend=2)
plt.title('Random Forest')
plt.show()


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Creazione e addestramento del modello Gradient Boosting
GradientBoosting_classification = GradientBoostingClassifier(random_state=5)
GradientBoosting_classification.fit(X_train, y_train)

# Inizializzazione delle liste se non sono già definite
model_classification = []
score_classification = []
recall_classification = []
precision_classification = []

# Calcolo delle metriche di valutazione
y_pred_GradientBoosting = GradientBoosting_classification.predict(X_test)
score_GradientBoosting = accuracy_score(y_test, y_pred_GradientBoosting)
recall_GradientBoosting = recall_score(y_test, y_pred_GradientBoosting)
precision_GradientBoosting = precision_score(y_test, y_pred_GradientBoosting, zero_division=0)

# Aggiunta del modello alla lista dei modelli
model_classification.append('Gradient Boosting')

# Aggiunta del punteggio di accuratezza alla lista dei punteggi
score_classification.append(score_GradientBoosting)

# Aggiunta del punteggio di richiamo alla lista dei punteggi
recall_classification.append(recall_GradientBoosting)

# Aggiunta del punteggio di precisione alla lista dei punteggi
precision_classification.append(precision_GradientBoosting)

# Stampa dei risultati
print("Modello:", model_classification[0])
print("Accuratezza:", score_classification[0])
print("Recall:", recall_classification[0])
print("Precisione:", precision_classification[0])

# Visualizzazione delle regioni decisionali
plot_decision_regions(X_test, y_test, clf=GradientBoosting_classification, legend=2)
plt.title('Gradient Boosting')
plt.show()



from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Creazione e addestramento del modello AdaBoostClassifier
AdaBoost_classification = AdaBoostClassifier(random_state=5)
AdaBoost_classification.fit(X_train, y_train)

# Inizializzazione delle liste se non sono già definite
model_classification = []
score_classification = []
recall_classification = []
precision_classification = []

# Calcolo delle metriche di valutazione
y_pred_AdaBoost = AdaBoost_classification.predict(X_test)
score_AdaBoost = accuracy_score(y_test, y_pred_AdaBoost)
recall_AdaBoost = recall_score(y_test, y_pred_AdaBoost)
precision_AdaBoost = precision_score(y_test, y_pred_AdaBoost, zero_division=0)

# Aggiunta del modello alla lista dei modelli
model_classification.append('AdaBoostClassifier')

# Aggiunta del punteggio di accuratezza alla lista dei punteggi
score_classification.append(score_AdaBoost)

# Aggiunta del punteggio di richiamo alla lista dei punteggi
recall_classification.append(recall_AdaBoost)

# Aggiunta del punteggio di precisione alla lista dei punteggi
precision_classification.append(precision_AdaBoost)

# Stampa dei risultati
print("Modello:", model_classification[0])
print("Accuratezza:", score_classification[0])
print("Recall:", recall_classification[0])
print("Precisione:", precision_classification[0])

# Visualizzazione delle regioni decisionali
plot_decision_regions(X_test, y_test, clf=AdaBoost_classification, legend=2)
plt.title('AdaBoostClassifier')
plt.show()


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Creazione e addestramento del modello SVC
SVC_classification = SVC(random_state=5)
SVC_classification.fit(X_train, y_train)

# Inizializzazione delle liste se non sono già definite
model_classification = []
score_classification = []
recall_classification = []
precision_classification = []

# Calcolo delle metriche di valutazione
y_pred_SVC = SVC_classification.predict(X_test)
score_SVC = accuracy_score(y_test, y_pred_SVC)
recall_SVC = recall_score(y_test, y_pred_SVC)
precision_SVC = precision_score(y_test, y_pred_SVC, zero_division=0)

# Aggiunta del modello alla lista dei modelli
model_classification.append('SVC')

# Aggiunta del punteggio di accuratezza alla lista dei punteggi
score_classification.append(score_SVC)

# Aggiunta del punteggio di richiamo alla lista dei punteggi
recall_classification.append(recall_SVC)

# Aggiunta del punteggio di precisione alla lista dei punteggi
precision_classification.append(precision_SVC)

# Stampa dei risultati
print("Modello:", model_classification[0])
print("Accuratezza:", score_classification[0])
print("Recall:", recall_classification[0])
print("Precisione:", precision_classification[0])

# Visualizzazione delle regioni decisionali
plot_decision_regions(X_test, y_test, clf=SVC_classification, legend=2)
plt.title('Support Vector Classifier (SVC)')
plt.show()

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Creazione e addestramento del modello Gaussian Naive Bayes
NB_classification = GaussianNB()
NB_classification.fit(X_train, y_train)

# Inizializzazione delle liste se non sono già definite
model_classification = []
score_classification = []
recall_classification = []
precision_classification = []

# Calcolo delle metriche di valutazione
y_pred_NB = NB_classification.predict(X_test)
score_NB = accuracy_score(y_test, y_pred_NB)
recall_NB = recall_score(y_test, y_pred_NB)
precision_NB = precision_score(y_test, y_pred_NB)

# Aggiunta del modello alla lista dei modelli
model_classification.append('Naive Bayes Gaussiano')

# Aggiunta del punteggio di accuratezza alla lista dei punteggi
score_classification.append(score_NB)

# Aggiunta del punteggio di richiamo alla lista dei punteggi
recall_classification.append(recall_NB)

# Aggiunta del punteggio di precisione alla lista dei punteggi
precision_classification.append(precision_NB)

# Stampa dei risultati
print("Modello:", model_classification[0])
print("Accuratezza:", score_classification[0])
print("Recall:", recall_classification[0])
print("Precisione:", precision_classification[0])

# Visualizzazione delle regioni decisionali
plot_decision_regions(X_test, y_test, clf=NB_classification, legend=2)
plt.title('Naive Bayes Gaussiano')
plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Creazione e addestramento del classificatore KNN
KNN_classifier = KNeighborsClassifier()
KNN_classifier.fit(X_train, y_train)

# Inizializzazione delle liste per memorizzare le metriche di valutazione
model_classification = []
score_classification = []
recall_classification = []
precision_classification = []

# Calcolo delle metriche di valutazione
y_pred_KNN = KNN_classifier.predict(X_test)
score_KNN = accuracy_score(y_test, y_pred_KNN)
recall_KNN = recall_score(y_test, y_pred_KNN)
precision_KNN = precision_score(y_test, y_pred_KNN)

# Aggiunta del modello alla lista dei modelli
model_classification.append('K-nearest Neighbors (KNN)')

# Aggiunta del punteggio di accuratezza alla lista dei punteggi
score_classification.append(score_KNN)

# Aggiunta del punteggio di richiamo alla lista dei punteggi
recall_classification.append(recall_KNN)

# Aggiunta del punteggio di precisione alla lista dei punteggi
precision_classification.append(precision_KNN)

# Stampa dei risultati
print("Modello:", model_classification[0])
print("Accuratezza:", score_classification[0])
print("Recall:", recall_classification[0])
print("Precisione:", precision_classification[0])

# Visualizzazione delle regioni decisionali
plot_decision_regions(X_test, y_test, clf=KNN_classifier, legend=2)
plt.title('K-nearest Neighbors (KNN)')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Lista dei modelli
models = ['Perceptron', 'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'AdaBoost', 'SVC', 'Naive Bayes', 'KNN']

# Liste delle metriche di valutazione per ogni modello
accuracies = [score_Perceptron, score_Logistic_Regression, score_RandomForest, score_GradientBoosting, score_AdaBoost, score_SVC, score_NB, score_KNN]
recalls = [recall_Perceptron, recall_Logistic_Regression, recall_RandomForest, recall_GradientBoosting, recall_AdaBoost, recall_SVC, recall_NB, recall_KNN]
precisions = [precision_Perceptron, precision_Logistic_Regression, precision_RandomForest, precision_GradientBoosting, precision_AdaBoost, precision_SVC, precision_NB, precision_KNN]

# Trova il modello migliore per ciascuna metrica
best_accuracy_model = models[np.argmax(accuracies)]
best_recall_model = models[np.argmax(recalls)]
best_precision_model = models[np.argmax(precisions)]

# Lista dei valori per il modello migliore
best_accuracy_value = np.max(accuracies)
best_recall_value = np.max(recalls)
best_precision_value = np.max(precisions)

# Plot dei valori per il modello migliore
plt.figure(figsize=(10, 6))
plt.bar(['Accuratezza', 'Recall', 'Precisione'], [best_accuracy_value, best_recall_value, best_precision_value], color=['blue', 'orange', 'green'])
plt.title('Performance del miglior modello ({})'.format(best_accuracy_model))
plt.ylabel('Metriche')
plt.xlabel('Valori')
plt.show()


# Calcola la media delle metriche per ciascun modello
mean_scores = [(score + recall + precision) / 3 for score, recall, precision in zip(accuracies, recalls, precisions)]

# Trova l'indice del modello con la media più alta
best_model_index = np.argmax(mean_scores)

# Ottieni il nome del modello migliore e la sua media
best_model = models[best_model_index]
best_model_mean_score = mean_scores[best_model_index]

print("Modello migliore (basato sulla media):", best_model)
print("Media delle metriche:", best_model_mean_score)

# Calcola la media delle metriche per ciascun modello
mean_scores = [(score + recall + precision) / 3 for score, recall, precision in zip(accuracies, recalls, precisions)]

# Trova l'indice del modello con la media più alta
best_model_index = np.argmax(mean_scores)

# Ottieni il nome del modello migliore e la sua media
best_model = models[best_model_index]
best_model_mean_score = mean_scores[best_model_index]

print("Modello migliore (basato sulla media):", best_model)
print("Media delle metriche:", best_model_mean_score)


import seaborn as sns

# Plot della matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_NB, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Naive Bayes Model')
plt.show()
