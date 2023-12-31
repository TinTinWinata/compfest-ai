import io
import os
import subprocess

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from mlxtend.plotting import plot_confusion_matrix
from nolds import corr_dim, dfa
from pyrpde import rpde
from sklearn import svm
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ParkinsonModel:
    def __init__(self):
        print('Building Parkinson Model...')
        self.check_model_exist()
        pass
        print('Parkinson Model[Done]')


    def check_model_exist(self):
        path = './model/parkinson.joblib'
        if(os.path.exists(path)):
            self.load_model()
        else:
            self.load_dataset()
            self.build_model()
            self.save_model()

    def load_dataset(self):
        self.read_dataset()
        self.preprocess()
    
    def read_dataset(self):
        self.df = pd.read_csv('./dataset/parkinsons.csv')
    
    def preprocess(self):

        # Drop Null Values 
        self.df.dropna(inplace=True)

        # Drop Duplicate Values
        self.df.drop_duplicates(inplace=True)


    def split_dataset(self):
        # Spliting dataset
        X = self.df.drop(columns=['name', 'status'], axis = 1)
        Y = self.df['status']

        # Split into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        return X_train, X_test, Y_train, Y_test

    def build_model(self):
        self.model = svm.SVC(kernel='linear')

        X_train, X_test, Y_train, Y_test = self.split_dataset()

        # Data standarization
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
        self.model.fit(X_train, Y_train)

        predictions = self.model.predict(X_test)

        cm = confusion_matrix(Y_test, predictions)

        fix, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True)

        plt.savefig('./report/parkinson.jpg')

        self.classification_report(Y_test, predictions)
        self.createRocAucReport(X_test, Y_test)

    def createRocAucReport(self, X_test, Y_test):
        y_scores = self.model.decision_function(X_test)

        roc_auc = roc_auc_score(Y_test, y_scores)

        print("ROC AUC Score:", roc_auc)

        fpr, tpr, thresholds = roc_curve(Y_test, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig('./report/parkinson_roc_auc.jpg')

    def classification_report(self, Y_test, predictions):
        acc = accuracy_score(Y_test, predictions)

        classificationreport = classification_report(Y_test, predictions)

        with open('./report/parkinson.txt', 'w') as f:
            f.write("Parkinson Prediction DiagnoAI.")
            f.write('\nAlgorithm : Support Vector Machine')
            f.write('\nAccuracy : ' + str(round(acc * 100, 2)))
            f.write('\nClassification_report : ')
            f.write(classificationreport)
    
    def load_model(self):
        self.model = joblib.load("./model/parkinson.joblib")

    def save_model(self):
        joblib.dump(self.model, "./model/parkinson.joblib", compress=3)
    
    # def predict(self, average_pitch, max_pitch, min_pitch, mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp,
    # mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq,
    # shimmer_dda, nhr, hnr, rpde_value, dfa_value, spread1, spread2, d2_value,
    # ppe):

    def convert_webm_to_mp3(self, input_path, output_path):
        try:
            subprocess.run(['ffmpeg', '-i', input_path, output_path])
            print("Conversion completed successfully.")
        except Exception as e:
            print("An error occurred:", str(e))


    def predict(self, data):
        
        audio_file = data 
        
        audio_signal, _ = librosa.load(audio_file)

        print(audio_signal)

        f0, voiced_flag, voiced_probs = librosa.pyin(audio_signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        voiced_f0 = f0[voiced_flag]

        average_pitch = np.mean(voiced_f0)
        max_pitch = np.max(voiced_f0)
        min_pitch = np.min(voiced_f0)

        jitter = np.diff(voiced_f0)

        mdvp_jitter_percent = np.mean(jitter) * 100
        mdvp_jitter_abs = np.mean(np.abs(jitter))
        mdvp_rap = np.mean(np.abs(np.diff(jitter)))
        mdvp_ppq = np.mean(np.abs(np.diff(jitter, n=2)))
        jitter_ddp = np.mean(np.abs(np.diff(jitter, n=1)))

        rms = librosa.feature.rms(y=audio_signal)[0]

        shimmer = np.diff(rms)

        mdvp_shimmer = np.mean(np.abs(shimmer))
        mdvp_shimmer_db = 10 * np.log10(np.mean(np.abs(shimmer)))
        shimmer_apq3 = np.mean(np.abs(np.diff(shimmer, n=2)))
        shimmer_apq5 = np.mean(np.abs(np.diff(shimmer, n=4)))
        mdvp_apq = np.mean(np.abs(np.diff(rms)))
        shimmer_dda = np.mean(np.abs(shimmer))

        harmonic_signal, percussive_signal = librosa.effects.hpss(audio_signal)

        nhr = np.sum(percussive_signal**2) / np.sum(harmonic_signal**2)

        hnr = np.sum(harmonic_signal**2) / np.sum(percussive_signal**2)

        normalized_pitch = (voiced_f0 - np.min(voiced_f0)) / (np.max(voiced_f0) - np.min(voiced_f0))

        normalized_audio_signal = audio_signal / np.max(np.abs(audio_signal))

        rpde_value = rpde(normalized_audio_signal, tau=30, dim=4, epsilon=0.01, tmax=1500)

        d2_value = corr_dim(normalized_pitch, emb_dim=5)

        normalized_pitch = (voiced_f0 - np.min(voiced_f0)) / (np.max(voiced_f0) - np.min(voiced_f0))

        dfa_value = dfa(normalized_pitch)

        spread1 = np.std(voiced_f0)

        spread2 = np.percentile(voiced_f0, 75) - np.percentile(voiced_f0, 25)

        hist, _ = np.histogram(voiced_f0, bins='auto')
        hist = hist / np.sum(hist)  
        ppe = -np.sum(hist * np.log2(hist + 1e-6))  
        
        data = pd.DataFrame({
            'Average Pitch (Fundamental Frequency)': [average_pitch],
            'Maximum Pitch (Fundamental Frequency)': [max_pitch],
            'Minimum Pitch (Fundamental Frequency)': [min_pitch],
            'MDVP:Jitter(%)' : [mdvp_jitter_percent],
            'MDVP:Jitter(Abs)' : [mdvp_jitter_abs],
            'MDVP:RAP' : [mdvp_rap],
            'MDVP:PPQ' : [mdvp_ppq],
            'Jitter:DDP' : [jitter_ddp],
            'MDVP:Shimmer' : [mdvp_shimmer],
            'MDVP:Shimmer(dB)' : [mdvp_shimmer_db],
            'Shimmer:APQ3' : [shimmer_apq3],
            'Shimmer:APQ5' : [shimmer_apq5],
            'MDVP:APQ' : [mdvp_apq],
            'Shimmer:DDA' : [shimmer_dda],
            'NHR' : [nhr],
            'HNR' : [hnr],
            'RPDE' : [rpde_value[0]],
            'DFA' : [dfa_value],
            'spread1' : [spread1],
            'spread2' : [spread2],
            'D2' : [d2_value],
            'PPE' : [ppe]
        })

        prediction = self.model.predict(data)

        return prediction