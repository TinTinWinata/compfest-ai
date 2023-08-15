import math
import os
import warnings
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from mlxtend.plotting import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from tensorflow.keras.models import Sequential


class CategoricalMapping:
    def __init__(self, map, name):
        self.map = map
        self.name = name

    def get_list(self):
        return [key for key in self.map]


class ANN_Model:
    def __init__(self, name, result_name, map_list, drop_list):
        print(f'Building {name.title()} Model...')
        self.drop_list: List[str] = drop_list
        self.map_list: List[CategoricalMapping] = map_list
        self.result_name: str = result_name
        self.name: str = name
        self.path = f'./model/{name}.joblib'
        self.dataset = f'./dataset/{name}.csv'
        self.report = f'./report/{name}.txt'
        self.scalar_path = f'./model/${name}-scalar.joblib'
        self.confusion = f'./report/{name}-confusion.jpg'
        self.check_model_exist()
        # self.print_df()
        print(f'{name.title()} Model [Done]')
        pass

    def check_model_exist(self):
        if(os.path.exists(self.path)):
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
        self.df = pd.read_csv(self.dataset)

    def map_categorical(self, name, map, expected_field):
        if expected_field:
            unique_field = self.df[name].unique()
            invalid_field = [
                field for field in unique_field if field not in expected_field]
            if invalid_field:
                print(f'Invalid field in {name} : {invalid_field}')
        self.df[name] = self.df[name].map(map)

    def print_df(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(self.df.head())

    def preprocess(self):
        # Drop Unused Columns
        self.df.drop(self.drop_list, axis=1, inplace=True)

        # Drop Duplicate Values
        self.df.drop_duplicates(inplace=True)

        # Custom Encoding
        for list in self.map_list:
            self.map_categorical(list.name, list.map, list.get_list())

        # Drop Null Values
        self.df.dropna(inplace=True)

        # Resamble Unbalanced Data
        class_0 = self.df[self.df[self.result_name] == 0]
        class_1 = self.df[self.df[self.result_name] == 1]

        # Over Sampling
        over_sample = class_1.sample(len(class_0), replace=True)

        # Concat Dataframe
        self.df = pd.concat([over_sample, class_0], axis=0)

    def split_dataset(self):
        # Split features and labels
        X = self.df.drop(self.result_name, axis=1)
        y = self.df[[self.result_name]]

        # Feature Scaling
        self.scalar = StandardScaler()
        X = self.scalar.fit_transform(X)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10)

        return X_train, X_test, y_train, y_test

    def build_model(self):
        # Initiate Sequntial model
        self.model = tf.keras.models.Sequential()

        # First Layer
        # self.model()

        X_train, X_test, y_train, y_test = self.split_dataset()

        self.model.fit(X_train, y_train.values.ravel())

        predictions = self.model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)

        fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True,
                                        show_normed=True,
                                        colorbar=True)

        plt.savefig(self.confusion)

        # test set results

        y_pred = self.model.predict(X_test)
        y_pred = np.array(y_pred)
        X_test = np.array(X_test)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        # print('Test : ', np.concatenate((y_pred.reshape(len(y_pred),1), X_test.reshape(len(X_test),len(X_test[0]))),1))

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

        with open(self.report, 'w') as f:
            f.write(f"--- {self.name.title()} Prediction ---")
            f.write('\nAlgorithm : Random Forest Classifier')
            f.write('\nAccuracy : ' +
                    str(round(acc*100, 2)))
            f.write('\nClassification_report : ')
            f.write(classificationreport)
            f.write('\nMean squared error : ' + str(mse))
            f.write('\nRoot mean squared error : ' + str(rmse))

    def load_model(self):
        self.model = joblib.load(self.path)

    def save_model(self):
        joblib.dump(self.model, self.path, compress=3)
        joblib.dump(self.scalar, self.scalar_path, compress=3)

    def predict(self, data):
        data = pd.DataFrame(data)
        scaler = joblib.load(self.scalar_path)
        data = scaler.transform(data)
        prediction = self.model.predict(data)
        return prediction
