import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.state_ops import scatter_nd_sub

NUM_CLASSES = 7
INPUT_SHAPE = (75, 100, 3)
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     except RuntimeError as e:
#         print(e)
#

def read_data():
    return pd.read_csv('./dcnn-dataset/HAM10000_metadata.csv')

def preprocess_data(df):
    # Preprocess Data
    paths = {}
    # Get a dictionary filled with all image Ids, and their path
    for x in glob(os.path.join("./dcnn-dataset", "*", "*.jpg")):
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
        'df': 'Dermatofibroma'
    }

    df['cell_type'] = df['dx'].map(lesion_type.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
    df['image'] = df['path'].map(lambda image: np.asarray(Image.open(image).resize((100, 75))))
    return df


def smote_sampling(features, target):
    rus = SMOTE(random_state=42)
    features, target = rus.fit_resample(features.reshape(-1, INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]), target)
    features = features.reshape(-1, *INPUT_SHAPE)
    return features, target

def image_preprocessing(features, target):
    features = np.asarray(features.tolist())
    feature_mean = np.mean(features)
    feature_std = np.std(features)

    print("Mean: ", feature_mean)
    print("Std: ", feature_std)

    features = (features - feature_mean) / feature_std

    # One Hot Encoder
    target = to_categorical(target, num_classes=NUM_CLASSES)

    # Reshape image to 3 dimension (75 * 100 * 3)
    features = features.reshape(features.shape[0], *INPUT_SHAPE)
    return smote_sampling(features, target)


def convolutional_block(x, growth_rate, dropout_rate=None):
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


def transition_block(x, reduction):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(x.shape[-1] * reduction), (1, 1), padding='same')(x)
    x = AveragePooling2D((2, 2), padding='same', strides=2)(x)
    return x


def dense_block(x, growth_rate, n_layers):
    for i in range(n_layers):
        x = convolutional_block(x, growth_rate)
    return x


# def DenseNet(growth_rate=32):
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


def residual_block(x, filters, down_sample=None):
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

# def ResNet50():
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
#
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

def ResNet_18():
    x_input = Input(INPUT_SHAPE)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same")(x_input)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)

    x = residual_block(x, filters=128, down_sample=True)
    x = residual_block(x, filters=128)

    x = residual_block(x, filters=256, down_sample=True)
    x = residual_block(x, filters=256)

    x = residual_block(x, filters=512, down_sample=True)
    x = residual_block(x, filters=512)

    x = GlobalAveragePooling2D()(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=x)
    return model


# def basic_model():
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


def make_model():
    model = ResNet_18()
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    model.build(input_shape=(None, *INPUT_SHAPE))
    print(model.summary())
    return model


def splitting_dataset(features, target):
    x_train, x_temp, y_train, y_temp = train_test_split(features, target, test_size=0.4, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    return x_train, x_val, x_test, y_train, y_val, y_test


def fitting_the_model(model, x_train, x_val, x_test, y_train, y_val, y_test):
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

    augmented_images, _ = next(datagen.flow(x_train, y_train, batch_size=10))

    fig, axes = plt.subplots(1, 10, figsize=(20, 20),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        ax.imshow(np.clip(np.squeeze(augmented_images[i]), 0, 255), cmap='gray')

    plt.show()

    # Fitting the model
    epochs = 1  # For temporary 10 so I can debug better
    batch_size = 5
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        verbose=1,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        callbacks=[learning_rate_reduction])
    plot_result(history)
    model.save("model/dcnn", save_format="tf")
    model.save("model/dcnn.h5", save_format="h5")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    loss_v, accuracy_v = model.evaluate(x_val, y_val, verbose=1)
    print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
    print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
    print_confusion_matrix(model, x_test, y_test)


def print_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 12))

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


def plot_result(history):
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


def main():
    skin_df = read_data()
    skin_df = preprocess_data(skin_df)

    features = skin_df["image"]
    target = skin_df["cell_type_idx"]
    features, target = image_preprocessing(features, target)
    x_train, x_val, x_test, y_train, y_val, y_test = splitting_dataset(features, target)

    print(features[:5])
    model = make_model()
    history = fitting_the_model(model, x_train, x_val, x_test, y_train, y_val, y_test)
    # model = tf.keras.models.load_model('dcnn.h5')
    # print_confusion_matrix(model=model,x_test=x_test, y_test=y_test)

if __name__ == "__main__":
    main()
