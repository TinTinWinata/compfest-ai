import os
from glob import glob
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tensorflow as tf
import tensorflow.python.keras.backend as K
from imblearn.over_sampling import SMOTE
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, MaxPool2D, MaxPooling2D,
                          ZeroPadding2D, concatenate)
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.state_ops import scatter_nd_sub

NUM_CLASSES = 8
INPUT_SHAPE = (75, 100, 3)


class DCNN_Model:
    def __init__(self):
        print("Available devices:", tf.config.list_physical_devices())
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        self.path = 'model/dcnn.h5'
        self.check_model_exist()
        self.mean = 159.8174582437455
        self.std = 46.41408053415013

    def check_model_exist(self):
        if os.path.exists(self.path):
            # If Exist Load Model
            print('Load Model')
            self.load_model()
        else:
            # If it doesn't Exit Build Model
            print('Making Model')
            self.main()

    # ----------------
    # MAKE MODEL
    # ----------------

    def read_data(self, ):
        return pd.read_csv('./dcnn-dataset/skin-lesion/HAM10000_metadata.csv')

    def preprocess_data(self, df):
        # Preprocess Data
        paths = {}

        # Get a dictionary filled with all image Ids, and their path
        for x in glob(os.path.join("./dcnn-dataset/skin-lesion/", "*", "*.jpg")):
            paths[os.path.splitext(os.path.basename(x))[0]] = x

        df['path'] = df['image_id'].map(paths.get)

        # Get the legible lesion type from the symbol
        lesion_type = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma',
            "clr": "Clear Skin"
        }

        df['cell_type'] = df['dx'].map(lesion_type.get)
        df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
        df['image'] = df['path'].map(lambda image: np.asarray(Image.open(image).resize((100, 75))))
        df = self.load_clear_skin_datasets(df)
        return df

    def smote_sampling(self, features, target):
        rus = SMOTE(random_state=42)
        features, target = rus.fit_resample(features.reshape(-1, INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]),
                                            target)
        features = features.reshape(-1, *INPUT_SHAPE)
        return features, target

    def image_preprocessing(self, features, target):
        features = np.asarray(features.tolist())
        feature_mean = np.mean(features)
        feature_std = np.std(features)

        print("Mean: ", feature_mean)
        print("Std: ", feature_std)

        features = (features - feature_mean) / feature_std

        # Blue starts here
        # plt.imshow((features[0]).astype('uint8'))
        # plt.title("Normalized Image")
        # plt.show()

        # One Hot Encoder
        target = to_categorical(target, num_classes=NUM_CLASSES)

        # Reshape image to 3 dimension (75 * 100 * 3)
        features = features.reshape(features.shape[0], *INPUT_SHAPE)
        return self.smote_sampling(features, target)

    def load_clear_skin_datasets(self, df):
        print("Loading Normal Clear Skin Dataset")

        clear_skin_image_list = []

        for filename in os.listdir("./dcnn-dataset/normal-skin"):
            path = "./dcnn-dataset/normal-skin/" + filename
            clear_skin_image_list.append(np.asarray(Image.open(path).resize((100, 75))))

        image_series = pd.Series(clear_skin_image_list, name='image')

        df = pd.concat([df, pd.DataFrame({"image": image_series, 'cell_type_idx': 7})], ignore_index=True)

        return df

    def convolutional_block(self, x, growth_rate, dropout_rate=None):
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(growth_rate * 4, (1, 1), padding='same')(x1)

        if dropout_rate:
            x1 = Dropout(dropout_rate)(x1)

        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = ZeroPadding2D((1, 1))(x1)
        x = ZeroPadding2D((1, 1))(x)
        x1 = Conv2D(growth_rate, (3, 3), padding='same')(x1)

        if dropout_rate:
            x1 = Dropout(dropout_rate)(x1)
        x = concatenate([x, x1])
        return x

    def transition_block(self, x, reduction):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(int(x.shape[-1] * reduction), (1, 1), padding='same')(x)
        x = AveragePooling2D((2, 2), padding='same', strides=2)(x)
        return x

    def dense_block(self, x, growth_rate, n_layers):
        for i in range(n_layers):
            x = self.convolutional_block(x, growth_rate)
        return x

    # def DenseNet(self, growth_rate=32):
    #     # Laptop ga kuat
    #     x_input = Input(INPUT_SHAPE)
    #
    #     x = Conv2D(growth_rate * 2, (7, 7), padding='same', strides=(2, 2))(x_input)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     x = AveragePooling2D((3, 3), padding='same', strides=2)(x)
    #
    #     x = dense_block(x, growth_rate, 4)
    #     x = transition_block(x, 0.5)
    #
    #     x = dense_block(x, growth_rate, 8)
    #     x = transition_block(x, 0.5)
    #
    #     x = dense_block(x, growth_rate, 16)
    #     x = transition_block(x, 0.5)
    #
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(NUM_CLASSES, activation='softmax')(x)
    #
    #     model = Model(inputs=x_input, outputs=x)
    #
    #     return model

    def residual_block(self, x, filters, down_sample=None):
        x_shortcut = x
        if down_sample:
            stride = 2  # The more stride, the smaller the picture
            x_shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='valid')(x_shortcut)
            x_shortcut = BatchNormalization()(x_shortcut)
        else:
            stride = 1

        x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x_shortcut, x])
        x = Activation('relu')(x)
        return x

    # def ResNet50(self, ):
    #     x_input = Input(INPUT_SHAPE)
    #     x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x_input)
    #     x = BatchNormalization()(x)
    #
    #     x = Activation('relu')(x)
    #     x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #
    #     x = residual_block(x, filters=64)
    #     x = residual_block(x, filters=64)
    #     x = residual_block(x, filters=64)


    #     x = residual_block(x, filters=128, down_sample=True)
    #     x = residual_block(x, filters=128)
    #     x = residual_block(x, filters=128)
    #     x = residual_block(x, filters=128)
    #
    #     x = residual_block(x, filters=256, down_sample=True)
    #     x = residual_block(x, filters=256)
    #     x = residual_block(x, filters=256)
    #     x = residual_block(x, filters=256)
    #     x = residual_block(x, filters=256)
    #     x = residual_block(x, filters=256)
    #
    #     x = residual_block(x, filters=512, down_sample=True)
    #     x = residual_block(x, filters=512)
    #     x = residual_block(x, filters=512)
    #
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(NUM_CLASSES, activation='softmax')(x)
    #
    #     return Model(inputs=x_input, outputs=x)

    def ResNet_18(self, ):
        x_input = Input(INPUT_SHAPE)
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same")(x_input)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.residual_block(x, filters=64)
        x = self.residual_block(x, filters=64)

        x = self.residual_block(x, filters=128, down_sample=True)
        x = self.residual_block(x, filters=128)
        x = Dropout(0.5)

        x = self.residual_block(x, filters=256, down_sample=True)
        x = self.residual_block(x, filters=256)
        x = Dropout(0.5)

        x = self.residual_block(x, filters=512, down_sample=True)
        x = self.residual_block(x, filters=512)
        x = Dropout(0.5)

        x = GlobalAveragePooling2D()(x)
        x = Dense(NUM_CLASSES, activation='softmax')(x)

        model = Model(inputs=x_input, outputs=x)
        return model

    # def basic_model(self, ):
    #     model = Sequential([
    #         Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=INPUT_SHAPE),
    #         Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
    #         MaxPool2D(pool_size=(2, 2)),
    #         Dropout(0.25),
    #
    #         Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", input_shape=INPUT_SHAPE),
    #         Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    #         MaxPool2D(pool_size=(2, 2)),
    #         Dropout(0.4),
    #
    #         Flatten(),
    #         Dense(128, activation="relu"),
    #         Dropout(0.5),
    #         Dense(NUM_CLASSES, activation="softmax")
    #     ])
    #     return model

    def make_model(self, ):
        model = self.ResNet_18()
        # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

        model.build(input_shape=(None, *INPUT_SHAPE))
        return model

    def splitting_dataset(self, features, target):
        x_train, x_temp, y_train, y_temp = train_test_split(features, target, test_size=0.4, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def fitting_the_model(self, model, x_train, x_val, x_test, y_train, y_val, y_test):
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    learning_rate=0.00001)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False  # randomly flip images
        )

        datagen.fit(x_train)
        print(np.min(x_train), np.max(x_train))
        augmented_images, _ = next(datagen.flow(x_train, y_train, batch_size=10))

        fig, axes = plt.subplots(1, 10, figsize=(20, 20),
                                 subplot_kw={'xticks': [], 'yticks': []},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))

        # for i, ax in enumerate(axes.flat):
        #     ax.imshow(np.clip(np.squeeze(augmented_images[i]), 0, 255), cmap='gray')

        # plt.show()
        print('Fitting Model ...')

        # Fitting the model
        epochs = 80
        batch_size = 32
        history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            verbose=1,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            callbacks=[learning_rate_reduction])
        self.plot_result(history)
        model.save("model/dcnn", save_format="tf")
        model.save("model/dcnn.h5", save_format="h5")

        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        loss_v, accuracy_v = model.evaluate(x_val, y_val, verbose=1)
        print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
        print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
        self.print_confusion_matrix(model, x_test, y_test)

    def print_confusion_matrix(self, model, x_test, y_test):
        y_pred = model.predict(x_test)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True,
                                        show_normed=True,
                                        colorbar=True)

        plt.savefig('./report/dcnn-confusion.jpg')

        lesion_type = [
            'Melanocytic nevi',
            'Melanoma',
            'Benign keratosis-like lesions ',
            'Basal cell carcinoma',
            'Actinic keratoses',
            'Vascular lesions',
            'Dermatofibroma'
        ]
        ax = sns.heatmap(cm, cmap="rocket_r", fmt=".01f", annot_kws={'size': 16}, annot=True, square=True,
                         xticklabels=lesion_type, yticklabels=lesion_type)  # What should I put here as a label
        ax.set_ylabel('Actual', fontsize=20)
        ax.set_xlabel('Predicted', fontsize=20)

    def plot_result(self, history):
        # Plot accuracy and loss
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label="Training accuracy")
        plt.plot(history.history['val_accuracy'], label="Validation accuracy")

        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label="Training loss")
        plt.plot(history.history['val_loss'], label="Validation loss")

        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig("Accuracy and Loss graph Perceptron.png")
        plt.show()

    def main(self):
        print('Reading Data')
        skin_df = self.read_data()
        print('Preprocess Data')
        skin_df = self.preprocess_data(skin_df)

        features = skin_df["image"]
        target = skin_df["cell_type_idx"]
        print('Image processing')
        features, target = self.image_preprocessing(features, target)
        print('Image Splitting Dataset')
        x_train, x_val, x_test, y_train, y_val, y_test = self.splitting_dataset(features, target)

        print(features[:5])
        print('Making model')
        self.model = self.make_model()
        self.fitting_the_model(self.model, x_train, x_val, x_test, y_train, y_val, y_test)

    # ----------------
    # LOAD MODEL
    # ----------------
    def load_model(self):
        self.model = load_model(self.path)

    def predict(self, image_path):
        input_image = self.preprocess_input_image(image_path)
        input_image = self.datagen_input_image(input_image)
        predict_probabilities = self.model.predict(np.expand_dims(input_image, axis=0), verbose=1)
        result = np.argmax(predict_probabilities)
        class_idx_to_label = {
            0: 'Melanocytic nevi',
            1: 'Melanoma',
            2: 'Benign keratosis-like lesions',
            3: 'Basal cell carcinoma',
            4: 'Actinic keratoses',
            5: 'Vascular lesions',
            6: 'Dermatofibroma',
            7: 'Clear Skin'
        }

        return {
            'predict': predict_probabilities.tolist()[0],
            'value': float(result),
            'result': class_idx_to_label[result]
        }

    def get_external_image(self, url):
        response = requests.get(url)
        return response.content

    def preprocess_input_image(self, image_path):
        img = Image.open(BytesIO(self.get_external_image(image_path)))
        img = np.array(img.resize((100, 75)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = np.array(img).astype('float64')
        img /= 255
        img = img.reshape(1, 75, 100, 1)
        # # Blue starts here
        # plt.imshow((img).astype('uint8'))
        # plt.title("Normalized Image")
        # plt.show()
        return img

    def datagen_input_image(self, input_image):
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
        )

        datagen.fit(input_image)
        augmented_iterator = datagen.flow(input_image, batch_size=1)
        augmented_image = next(augmented_iterator)[0]
        # plt.imshow(np.clip(np.squeeze(augmented_image), 0, 255), cmap='gray')
        # plt.show()

        return augmented_image
