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
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    

    def read_dataset(self):
        self.df = pd.read_csv(
            './dataset/stroke.csv')

    def map_categorical(self, name, map, expected_field):
        if expected_field:
            unique_field = self.df[name].unique()
            invalid_field = [field for field in unique_field if field not in expected_field]
            if invalid_field:
                print(f'Invalid field in {name} : {invalid_field}')
        self.df[name] = self.df[name].map(map)        


    def print_df(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(self.df.head())

    def preprocess(self):
        # Drop Unused Columns
        self.df.drop(['id'], axis=1, inplace=True)


        # Drop Duplicate Values
        self.df.drop_duplicates(inplace=True)

        # Initialize Label Encoder
        label_encoder = LabelEncoder()

        # Encode categorical
        # categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        # for col in categorical_columns:
        # self.df[col] = label_encoder.fit_transform(self.df[col])

        # Custom Encoding
        gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
        ever_married_mapping = {'No': 0, 'Yes': 1}
        work_type_mapping = {'children': 0, 'Govt_jov': 1, 'Never_worked': 2, 'Private': 3 , 'Self-employed': 4}
        residence_type_mapping = {'Rural': 0, 'Urban': 1}
        smoking_status_mapping = {'formerly smoked': 3, 'never smoked': 2, 'smokes': 1, 'Unknown': 0}

        self.map_categorical('gender', gender_mapping, ['Male', 'Female', 'Other'])
        self.map_categorical('ever_married', ever_married_mapping, ['No', 'Yes'])
        self.map_categorical('work_type', work_type_mapping, ['children', 'Govt_jov', 'Never_worked', 'Self-employed'])
        self.map_categorical('residence_type', residence_type_mapping, ['Rural', 'Urban'])
        self.map_categorical('smoking_status', smoking_status_mapping, ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

        # Drop Null Values
        self.df.dropna(inplace=True)

        self.df.to_csv('test.csv', index=False)

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

        # Feature Scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print(X)
                # self.print_df()

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
            'residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status],
        })

        prediction = self.model.predict(data)

        return prediction
