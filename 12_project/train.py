from pathlib import Path
import pandas as pd
import tensorflow.keras.applications.mobilenet_v2 as mn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf

''' Dataframe creation
'''
def proc_img(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
        Source: https://www.kaggle.com/databeru/fruit-and-vegetable-classification/
    """

    labels = [str(filepath[i]).split("/")[-2] \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop = True)
    
    return df

def create_dataframes():
    # Create a list with the filepaths for training and testing
    train_dir = Path('./data/train')
    train_filepaths = list(train_dir.glob(r'**/*.jpg'))

    val_dir = Path('./data/validation')
    val_filepaths = list(val_dir.glob(r'**/*.jpg'))

    test_dir = Path('./data/test')
    test_filepaths = list(test_dir.glob(r'**/*.jpg'))

    train_df = proc_img(train_filepaths)
    test_df = proc_img(test_filepaths)
    val_df = proc_img(val_filepaths)
    full_train_df = pd.concat([train_df, val_df])

    return train_df, test_df, full_train_df, val_df

def create_datasets(train_df, test_df, full_train_df, val_df):
    train_gen_mn = ImageDataGenerator(
        preprocessing_function=mn.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_ds_mn = train_gen_mn.flow_from_dataframe(
        dataframe= train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )

    val_gen_mn = ImageDataGenerator(
        preprocessing_function=mn.preprocess_input,
    )

    val_ds_mn = val_gen_mn.flow_from_dataframe(
        dataframe= val_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0
    )

    full_train_gen_mn = ImageDataGenerator(
        preprocessing_function=mn.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    full_train_ds_mn = full_train_gen_mn.flow_from_dataframe(
            dataframe= full_train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )

    test_gen_mn = ImageDataGenerator(
        preprocessing_function=mn.preprocess_input
    )

    test_ds_mn = test_gen_mn.flow_from_dataframe(
        dataframe= test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0
    )

    return train_ds_mn, val_ds_mn, full_train_ds_mn, test_ds_mn

''' Train
'''

def make_model():
    base_model = mn.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    x = keras.layers.Dense(128, activation='relu')(vectors)
    x = keras.layers.Dense(96, activation='relu')(x)
    outputs = keras.layers.Dense(36)(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss = keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )

    return model

def train_model(model, full_train_ds_mn, val_ds_mn):
    checkpoint = keras.callbacks.ModelCheckpoint(
        'mobilenet_final_retrained.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    history = model.fit(
        full_train_ds_mn,
        epochs=10,
        validation_data=val_ds_mn,
        callbacks=[checkpoint]
    )

    return model, history

''' main loop
'''
def main():
    # dataset loading
    train_df, test_df, full_train_df, val_df = create_dataframes()
    train_ds_mn, val_ds_mn, full_train_ds_mn, test_ds_mn  = create_datasets(train_df, test_df, full_train_df, val_df)
    # create and train model
    model = make_model()
    model, history = train_model(model, full_train_ds_mn, val_ds_mn)
    # test model
    model = keras.models.load_model('mobilenet_final_retrained.h5')
    model.evaluate(test_ds_mn)
    # convert to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_lite = converter.convert()
    with open('model_retrained.tflite', 'wb') as f_out:
        f_out.write(model_lite)

if __name__ == "__main__":
    main()