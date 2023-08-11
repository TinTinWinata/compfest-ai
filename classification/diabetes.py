import math
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error)
from sklearn.model_selection import train_test_split


class DiabetesModel:
    def __init__(self):

        self.check_model_exist()
        pass

    def check_model_exist(self):
        path = './model/diabetes.joblib'
        if(os.path.exists(path)):
            # If Exist Load Model
            self.load_model()
        else:
            # If Doesn't Exit Build Model
            self.load_dataset()
            self.build_model()
            self.save_model()

    def load_dataset(self):
        self.read_dataset()
        self.preprocess()

    def read_dataset(self):
        self.df = pd.read_csv(
            './dataset/diabetes.csv')

    def preprocess(self):
        # Drop Unused Columns
        self.df.drop(['Education', 'Income'], axis=1, inplace=True)

        # Drop Null Values
        self.df.dropna(inplace=True)

        # Drop Duplicate Values
        self.df.drop_duplicates(inplace=True)

        # Dataset balancing
        class_0 = self.df[self.df['Diabetes_binary'] == 0]
        class_1 = self.df[self.df['Diabetes_binary'] == 1]

        # Over Sampling
        over_sample = class_1.sample(len(class_0), replace=True)

        # Concat Dataframe
        self.df = pd.concat([over_sample, class_0], axis=0)

    def split_dataset(self):
        # Split features and labels
        X = self.df.drop('Diabetes_binary', axis=1)
        y = self.df[['Diabetes_binary']]

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10)

        return X_train, X_test, y_train, y_test

    def build_model(self):
        self.model = RandomForestClassifier(
            n_estimators=300, criterion='entropy', min_samples_split=10, random_state=0)

        X_train, X_test, y_train, y_test = self.split_dataset()

        self.model.fit(X_train, y_train.values.ravel())

        predictions = self.model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)

        fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True,
                                        show_normed=True,
                                        colorbar=True)

        plt.savefig('confusion_matrix.jpg')

        self.classification_report(y_test, predictions)

    def classification_report(self, y_test, predictions):
        # acc
        acc = accuracy_score(y_test, predictions)

        # calculating the classification report
        classificationreport = classification_report(y_test, predictions)

        # calculating the mse
        mse = mean_squared_error(y_test, predictions)

        # calculating the rmse
        rmse = math.sqrt(mse)

        with open('report.txt', 'w') as f:
            f.write("This message will be written to a file.")
            f.write('\nAccuracy score of Random Forest Classifier : ' +
                    str(round(acc*100, 2)))
            f.write('\nClassification_report : ')
            f.write(classificationreport)
            f.write('\nMean squared error : ' + str(mse))
            f.write('\nRoot mean squared error : ' + str(rmse))

    def load_model(self):
        self.model = joblib.load("./model/diabetes.joblib")

    def save_model(self):
        joblib.dump(self.model, "./model/diabetes.joblib", compress=3)

    def predict(self,
                high_bp, high_chol, chol_check, bmi, smoker, stroke,
                heart_disease, phys_activity, fruits, veggies, heavy_alc,
                health_insurance, no_doc_bc_cost, gen_health, mental_health,
                phys_health, diff_walk, sex, age_category):

        data = pd.DataFrame({
            'HighBP': [high_bp],
            'HighChol': [high_chol],
            'CholCheck': [chol_check],
            'BMI': [bmi],

            'Smoker': [smoker],
            'Stroke': [stroke],
            'HeartDiseaseorAttack': [heart_disease],
            'PhysActivity': [phys_activity],

            'Fruits': [fruits],
            'Veggies': [veggies],
            'HvyAlcoholConsump': [heavy_alc],
            'AnyHealthcare': [health_insurance],

            'NoDocbcCost': [no_doc_bc_cost],
            'GenHlth': [gen_health],
            'MentHlth': [mental_health],
            'PhysHlth': [phys_health],

            'DiffWalk': [diff_walk],
            'Sex': [sex],
            'Age': [age_category],
        })

        prediction = self.model.predict(data)

        return prediction
