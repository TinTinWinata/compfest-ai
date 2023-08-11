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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


class StrokeModel:
    def __init__(self):

        self.check_model_exist()
        pass

    def check_model_exist(self):
        path = './model/stroke.joblib'
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
        self.feature_scale()
    
    def feature_scale(self):
        self.df.head()
        # sc = StandardScaler()
        # print('test')

    def read_dataset(self):
        self.df = pd.read_csv(
            './dataset/stroke.csv')

    def preprocess(self):
        # Drop Unused Columns
        self.df.drop(['id'], axis=1, inplace=True)

        # Drop Null Values
        self.df.dropna(inplace=True)

        # Drop Duplicate Values
        self.df.drop_duplicates(inplace=True)

        # Encode categorical
        self.df = pd.get_dummies(self.df, columns=[
                                 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

        # Resamble Unbalanced Data
        class_0 = self.df[self.df['stroke'] == 0]
        class_1 = self.df[self.df['stroke'] == 1]

        # Over Sampling
        over_sample = class_1.sample(len(class_0), replace=True)

        # Concat Dataframe
        self.df = pd.concat([over_sample, class_0], axis=0)

    def split_dataset(self):
        # Split features and labels
        X = self.df.drop('stroke', axis=1)
        y = self.df[['stroke']]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

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
        self.model = joblib.load("./model/stroke.joblib")

    def save_model(self):
        joblib.dump(self.model, "./model/stroke.joblib", compress=3)

    def predict(self,
                gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):

        data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status],
        })

        prediction = self.model.predict(data)

        return prediction
