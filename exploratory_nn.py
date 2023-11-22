import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd


def count(df):
    counts = df.groupby(['label', 'hour']).size().reset_index(name='count')
    df = df.merge(counts, on=['label', 'hour'], how='left')
    return df


def save_count(path):
    df = pd.read_csv(f'data/{path}.csv')
    df = count(df)
    df = df.drop_duplicates()
    df.to_csv(f'data/{path}_count.csv', index=False)


def split_x_y(df):
    x = df.drop(['count'], axis=1)
    y = df['count']
    return x, y


def build_model():
    input = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(input)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    return model


def main():
    arrival_train = pd.read_csv('data/arrival_train_count.csv')
    arrival_test = pd.read_csv('data/arrival_test_count.csv')
    arrival_eval = pd.read_csv('data/arrival_eval_count.csv')
    departure_train = pd.read_csv('data/departure_train_count.csv')
    departure_test = pd.read_csv('data/departure_test_count.csv')
    departure_eval = pd.read_csv('data/departure_eval_count.csv')

    # xy
    x_train, y_train = split_x_y(arrival_train)
    x_test, y_test = split_x_y(arrival_test)
    x_eval, y_eval = split_x_y(arrival_eval)

    # model settings
    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.Accuracy()]

    # build model
    model = build_model()
    model.compile(optimizer=optimiser, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # evaluate
    model.evaluate(x_eval, y_eval)
    print(model.summary())
