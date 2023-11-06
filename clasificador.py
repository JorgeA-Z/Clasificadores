import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

class Clasificador:
    def __init__(self, tipo) -> None:
        self.dataset = None
        self.tipo = tipo
        pass

    def Logistic_Regression(self):
        match self.tipo:
            case 1:
                x = self.dataset.drop("Class variable (0 or 1)", axis=1)
                y = self.dataset["Class variable (0 or 1)"]

            case 2:
                x = self.dataset[['X']]

                y = self.dataset['Y']

            case 3:
                x = self.dataset.drop("quality", axis=1)
                y = self.dataset["quality"]
                #y = (y > 6).astype(int)


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    
        model = LogisticRegression()
        
        model.fit(x_train, y_train)
            
        # Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)


        # Visualizar los datos reales en un diagrama de dispersión
        if self.tipo == 1:
            plt.scatter(x_test["Number of times pregnant"], x_test["Age (years)"], c=y_pred, s=30)
            plt.xlabel('Number of times pregnant')
            plt.ylabel('Age (years)')
            

        elif self.tipo == 3:
            plt.scatter(x_test["fixed acidity"], x_test["alcohol"], c=y_pred, s=30)
            plt.xlabel('fixed acidity')
            plt.ylabel('alcohol')

        plt.title("Exactitud del modelo: {:.2f}".format(accuracy))
        plt.show()

        print("Informe de clasificación:")
        print(report)



    def K_Nearest_Neighbors(self):
        match self.tipo:
            case 1:
                x = self.dataset.drop("Class variable (0 or 1)", axis=1)
                y = self.dataset["Class variable (0 or 1)"]

            case 2:
                x = self.dataset[['X']]

                y = self.dataset['Y']

            case 3:
                x = self.dataset.drop("quality", axis=1)
                y = self.dataset["quality"]
        
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        k = 3

        model = KNeighborsClassifier(n_neighbors=k)
        
        model.fit(x_train, y_train)
        
        
        y_pred = model.predict(x_test)


        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        # Visualizar los datos reales en un diagrama de dispersión
        if self.tipo == 1:
            plt.scatter(x_test["Number of times pregnant"], x_test["Age (years)"], c=y_pred, s=30)
            plt.xlabel('Number of times pregnant')
            plt.ylabel('Age (years)')
            

        elif self.tipo == 3:
            plt.scatter(x_test["fixed acidity"], x_test["alcohol"], c=y_pred, s=30)
            plt.xlabel('fixed acidity')
            plt.ylabel('alcohol')

        plt.title("Exactitud del modelo: {:.2f}".format(accuracy))
        plt.show()

        print("Informe de clasificación:")
        print(report)


    def Support_Vector_Machines(self):
        match self.tipo:
            case 1:
                x = self.dataset.drop("Class variable (0 or 1)", axis=1)
                y = self.dataset["Class variable (0 or 1)"]

            case 2:
                x = self.dataset[['X']]

                y = self.dataset['Y']

            case 3:
                x = self.dataset.drop("quality", axis=1)
                y = self.dataset["quality"]
        
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = SVC(kernel='sigmoid', C=1.0)
        
        model.fit(x_train, y_train)
        
        
        y_pred = model.predict(x_test)


        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        # Visualizar los datos reales en un diagrama de dispersión
        if self.tipo == 1:
            plt.scatter(x_test["Number of times pregnant"], x_test["Age (years)"], c=y_pred, s=30)
            plt.xlabel('Number of times pregnant')
            plt.ylabel('Age (years)')
            

        elif self.tipo == 3:
            plt.scatter(x_test["fixed acidity"], x_test["alcohol"], c=y_pred, s=30)
            plt.xlabel('fixed acidity')
            plt.ylabel('alcohol')

        plt.title("Exactitud del modelo: {:.2f}".format(accuracy))
        plt.show()

        print("Informe de clasificación:")
        print(report)

    def Naive_Bayes(self):
        match self.tipo:
            case 1:
                x = self.dataset.drop("Class variable (0 or 1)", axis=1)
                y = self.dataset["Class variable (0 or 1)"]

            case 2:
                x = self.dataset[['X']]

                y = self.dataset['Y']

            case 3:
                x = self.dataset.drop("quality", axis=1)
                y = self.dataset["quality"]
        
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = MultinomialNB()
        
        model.fit(x_train, y_train)
        
        
        y_pred = model.predict(x_test)


        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        # Visualizar los datos reales en un diagrama de dispersión
        if self.tipo == 1:
            plt.scatter(x_test["Number of times pregnant"], x_test["Age (years)"], c=y_pred, s=30)
            plt.xlabel('Number of times pregnant')
            plt.ylabel('Age (years)')
            

        elif self.tipo == 3:
            plt.scatter(x_test["fixed acidity"], x_test["alcohol"], c=y_pred, s=30)
            plt.xlabel('fixed acidity')
            plt.ylabel('alcohol')

        plt.title("Exactitud del modelo: {:.2f}".format(accuracy))
        plt.show()

        print("Informe de clasificación:")
        print(report)

    def redNeuronal(self):
        match self.tipo:
            case 1:
                x = self.dataset.drop("Class variable (0 or 1)", axis=1)
                y = self.dataset["Class variable (0 or 1)"]

            case 2:
                x = self.dataset[['X']]

                y = self.dataset['Y']

            case 3:
                x = self.dataset.drop("quality", axis=1)
                y = self.dataset["quality"]
    
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)


        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        
        if self.tipo == 1:
            plt.scatter(x_test["Number of times pregnant"], x_test["Age (years)"], c=y_pred, s=30)
            plt.xlabel('Number of times pregnant')
            plt.ylabel('Age (years)')
            

        elif self.tipo == 3:
            plt.scatter(x_test["fixed acidity"], x_test["alcohol"], c=y_pred, s=30)
            plt.xlabel('fixed acidity')
            plt.ylabel('alcohol')

        plt.title("Exactitud del modelo: {:.2f}".format(accuracy))
        plt.show()

        print("Informe de clasificación:")
        print(report)

    def read_Dataset(self):
        #Diabetes
        if self.tipo == 1:
            self.dataset = pd.read_csv("Datasets/Pima_Indians_Diabetes_Dataset.csv", delimiter=",")
            
        #Seguranza de autos
        elif self.tipo == 2:
            self.dataset = pd.read_csv("Datasets/Swedish_Auto_Insurance_Dataset.csv", delimiter=",")
        
        #Calidad el vino
        elif self.tipo == 3:
            self.dataset = pd.read_csv("Datasets/Wine_Quality_Dataset.csv", delimiter=";")