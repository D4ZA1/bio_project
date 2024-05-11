 # This Python script loads gene expression data, preprocesses it, trains a neural network model, and evaluates it using cross-validation.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

features = ['FDR adjusted p-value', 'Cancer Sample Med', 'Normal Sample Med', 'log2 fold change']


def load_data(directory):
    all_data = []
    files = os.listdir(directory)

    for filename in tqdm(files, desc="Loading data"):
        if filename.endswith('_Differential_Gene_Expression_Table.txt'):
            df = pd.read_csv(os.path.join(directory, filename), sep='\t')
            df['Cancer type'] = filename.split('_')[0]

            all_data.append(df)
    all_data = pd.concat(all_data)

    X = all_data[features]
    y = all_data['Cancer type']

    return X, y


def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def create_model(input_dim, num_classes):
    Model = Sequential()
    Model.add(Dense(64, activation='relu', input_dim=input_dim))
    Model.add(Dense(64, activation='relu'))
    Model.add(Dense(num_classes, activation='softmax'))

    Model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return Model


def train_model(X_train, y_train):
    clf = MLPClassifier(random_state=42)
    pipeline = Pipeline(steps=[('s', StandardScaler()), ('m', clf)])
    grid = {'m__hidden_layer_sizes': [(50,), (100,), (150,)], 'm__activation': ['relu', 'tanh', 'logistic']}

    grid_search = GridSearchCV(estimator=pipeline, param_grid=grid, cv=5)

    print("Starting to fit the model...")
    grid_search.fit(X_train, y_train)
    print("Finished fitting the model.")


def train_model_in_epochs(X_train, y_train, epochs=10, batch_size=100):
    y_train_one_hot = to_categorical(y_train)
    input_dim = X_train.shape[1]
    num_classes = y_train_one_hot.shape[1]

    model = create_model(input_dim, num_classes)

    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}...")
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train_one_hot[start:end]

            # Fit the model on the batch
            print(f"Starting to fit the model on batch {i + 1}...")
            model.fit(X_batch, y_batch, epochs=1, verbose=0)
            print(f"Finished fitting the model on batch {i + 1}.")

    return model


def evaluate_model(model, X_test, y_test):
    y_test_one_hot = to_categorical(y_test)
    y_pred = model.predict(X_test)

    print('Accuracy:', accuracy_score(y_test_one_hot, y_pred))


if __name__ == "__main__":
    with tqdm(total=5, desc="Overall Progress") as pbar:
        print("Starting to load data...")
        X, y = load_data('./expression')
        print("Finished loading data.")
        pbar.update()

        print("Starting to split data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Finished splitting data.")
        pbar.update()

        print("Starting to preprocess data...")
        X_train, X_test = preprocess_data(X_train, X_test)
        print("Finished preprocessing data.")
        pbar.update()

        print("Starting to train model...")
        model = train_model_in_epochs(X_train, y_train)
        print("Finished training model.")
        pbar.update()

        print("Starting to evaluate model...")
        evaluate_model(model, X_test, y_test)
        print("Finished evaluating model.")
        pbar.update()

        print("Starting cross-validation...")
        cv_score = cross_val_score(model, X, y, cv=5)
        print('Cross-validation score:', cv_score.mean())
        print("Finished cross-validation.")
