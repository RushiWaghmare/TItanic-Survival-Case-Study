import math
import numpy as np
import pandas as pd
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def MarvellousTitanicLogistic():
    # Step 1: Load data
    titanic_data = pd.read_csv("MarvellousTitanicDataset.csv")
    print("Top 5 entries of dataset:")
    print(titanic_data.head())

    
    # Step 2: Analyse data
    # Survived
    print("Visualisation : Survived and non-survived passengers")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target).set_title("Marvellous Infosystem: Survived and non-survived passengers")
    show()

    # Gender
    print("Visualisation : Survived and non-survived passengers based on gender")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target, hue="Sex").set_title("Marvellous Infosystem: Survived and non-survived passengers based on gender")
    show()

    # Passenger Class
    print("Visualisation: Survived and non-survived passengers based on Passenger class")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target, hue="Pclass").set_title("Marvellous Infosystem: Survived and non-survived passengers based on Passenger class")
    show()

    # Age
    print("Visualisation: Survived and non-survived passengers based on Passenger Age")
    figure()
    titanic_data['Age'].plot.hist().set_title("Marvellous Infosystem: Survived and non-survived passengers based on Passenger Age")
    show()

    # Fare
    print("Visualisation: Survived and non-survived passengers based on Fare")
    figure()
    titanic_data['Fare'].plot.hist().set_title("Marvellous Infosystem: Survived and non-survived passengers based on Fare")
    show()
    

    # Step 3: Cleaning the data = "Zero" column
    titanic_data.drop("zero", axis=1, inplace=True)
    print("Top 5 entries of data after cleaning the data:")
    print(titanic_data.head(5))

    print("Values of Sex column:")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of Sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
    print(Sex.head(5))

    print("Values of Pclass column after removing one field")
    Pclass = pd.get_dummies(titanic_data['Pclass'], drop_first=True)
    print(Pclass.head(5))

    # Concatenate the Data
    print("Values of data set after concatenating new columns")
    titanic_data = pd.concat([titanic_data, Sex, Pclass], axis=1)
    print(titanic_data.head(5))

    # Remove irrelevant columns
    print("Values of data set after removing irrelevant columns")
    titanic_data.drop(["Sex", "SibSp", "Parch", "Embarked"], axis=1, inplace=True)
    print(titanic_data.head(5))

    # Convert all column names to strings to avoid errors
    titanic_data.columns = titanic_data.columns.astype(str)

    X = titanic_data.drop("Survived", axis=1)
    Y = titanic_data["Survived"]

    # Step 4: Data Train
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

    logmodel = LogisticRegression(max_iter=200)

    logmodel.fit(Xtrain, Ytrain)

    # Step 5: Data Testing
    prediction = logmodel.predict(Xtest)

    # Step 6: Accuracy of model
    result = accuracy_score(Ytest, prediction)
    print("Accuracy of Logistic Regression Algorithm is:", result * 100, "%")

def main():
    MarvellousTitanicLogistic()

if __name__ == "__main__":
    main()
