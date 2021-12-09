from pathlib import Path
import pandas as pd
import tensorflow.keras.applications.mobilenet_v2 as mn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

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
    train_dir = Path('./data/veggies/train')
    train_filepaths = list(train_dir.glob(r'**/*.jpg'))

    val_dir = Path('./data/veggies/validation')
    val_filepaths = list(val_dir.glob(r'**/*.jpg'))

    test_dir = Path('./data/veggies/test')
    test_filepaths = list(test_dir.glob(r'**/*.jpg'))

    train_df = proc_img(train_filepaths)
    test_df = proc_img(test_filepaths)
    val_df = proc_img(val_filepaths)

    return train_df, test_df, val_df

def create_datasets(train_df, val_df):
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
        target_size=(299, 299),
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
        target_size=(299, 299),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0
    )

    return train_ds_mn, val_ds_mn

''' Train
'''

def cmopose_final_model(base_model, input_shape=(299, 299, 3), learning_rate=1e-3):
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    dense1 = keras.layers.Dense(128, activation='relu')(vectors)
    dense2 = keras.layers.Dense(128, activation='relu')(dense1)
    outputs = keras.layers.Dense(36)(dense2)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss = keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )

    return model

def make_train_model(train_ds_mn, val_ds_mn, learning_rate = 0.001):
    base_model_mn = mn.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3)
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        'mobilenet_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    final_model_mn = cmopose_final_model(base_model=base_model_mn, learning_rate=learning_rate)

    history = final_model_mn.fit(
        train_ds_mn,
        epochs=20,
        validation_data=val_ds_mn,
        callbacks=[checkpoint]
    )

    return final_model_mn, history


''' Testing
'''
def evaluate_model(model, test_df):

    test_gen_mn = ImageDataGenerator(
        preprocessing_function=mn.preprocess_input
    )

    test_ds_mn = test_gen_mn.flow_from_dataframe(
        dataframe= test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(299, 299),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0
    )

    model.evaluate(test_ds_mn)

''' Save the model
'''
def save_model(model, model_path = 'model.bin'):
    model.save(model_path)

''' main loop
'''
def main():
    # dataset loading
    train_df, test_df, val_df = create_dataframes()
    train_ds_mn, val_ds_mn = create_datasets(train_df, val_df)
    # create and train model
    final_model_mn, history = make_train_model(train_ds_mn, val_ds_mn, learning_rate=0.001)
    # test model
    evaluate_model(final_model_mn, test_df)
    # save model
    save_model(final_model_mn, model_path='model.bin')

if __name__ == "__main__":
    main()