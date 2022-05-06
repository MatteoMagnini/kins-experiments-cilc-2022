from typing import Iterable
import numpy as np
import pandas as pd
from psyki.ski import Injector, Formula
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import sparse_categorical_crossentropy, sparse_categorical_accuracy
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense


def grid_search(data: pd.DataFrame,
                classes: int,
                layers: list[int] = [2, 3, 4, 5],
                neurons: list[int] = [16, 32, 64, 128],
                activations: list[str] = ['relu', 'tanh'],
                batches: list[int] = [16, 32, 64, 128],
                population_size: int = 30,
                epochs: int = 100,
                seed: int = 0,
                stop: EarlyStopping = None):

    best: tuple[int, dict] = 0, {}
    for activation in activations:
        for layer in layers:
            for neuron in neurons:
                for batch in batches:
                    val_acc = []
                    for i in range(0, population_size):
                        set_seed(seed+i)
                        predictor = generate_predictor(data.shape[1]-1, classes, layer, neuron, activation)
                        train, test = train_test_split(data, train_size=1000, random_state=seed+i, stratify=data.iloc[:, -1])
                        train, val = train_test_split(train, test_size=100, random_state=seed+i, stratify=train.iloc[:, -1])
                        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
                        val_x, val_y = val.iloc[:, :-1], val.iloc[:, -1:]
                        # test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]

                        predictor.fit(train_x, train_y, epochs=epochs,
                                      batch_size=batch, verbose=0, callbacks=stop)

                        loss, acc = predictor.evaluate(val_x, val_y)
                        val_acc.append(acc)
                    mean = sum(val_acc)/len(val_acc)
                    print('Mean accuracy with\nActivation = ' + activation + '\nBatch = ' + str(batch) + '\nLayers = ' + str(layer) + '\nNeurons = ' + str(neuron) + '\n> ' + str(mean) +'\n\n')
                    if best[0] < mean:
                        best = mean, {'activation': activation,
                                      'batch': batch,
                                      'layers': layer,
                                      'neurons': neuron}
    return best


def generate_predictor(input_size: int, output_size, layers: int, neurons: int, activation: str):
    inputs = Input((input_size,))
    x = Dense(neurons, activation=activation)(inputs)
    for _ in range(1, layers):
        x = Dense(neurons, activation=activation)(x)
    x = Dense(output_size, activation='softmax' if output_size > 1 else 'sigmoid')(x)
    predictor = Model(inputs, x)
    predictor.compile('adam', loss=('sparse_categorical_crossentropy' if output_size > 1 else 'binary_crossentropy'),
                      metrics=['accuracy'])
    return predictor


def run_experiments(data: pd.DataFrame,
                    injector: Injector,
                    knowledge: Iterable[Formula],
                    population_size: int = 30,
                    seed: int = 0,
                    epochs: int = 100,
                    batch_size: int = 32,
                    stop: EarlyStopping = None
                    ) -> pd.DataFrame:

    loss_k = []
    acc_k = []

    for i in range(0, population_size):
        seed = seed + i
        set_seed(seed)

        predictor_k = injector.inject(knowledge)

        predictor_k.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        train, test = train_test_split(data, train_size=1000, random_state=seed, stratify=data.iloc[:, -1])
        train, val = train_test_split(train, test_size=100, random_state=seed, stratify=train.iloc[:, -1])
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        val_x, val_y = val.iloc[:, :-1], val.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]

        predictor_k.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs,
                        batch_size=batch_size, verbose=0, callbacks=stop)
        l_k, a_k = predictor_k.evaluate(test_x, test_y)

        loss_k.append(l_k)
        acc_k.append(a_k)

        del predictor_k

    return pd.DataFrame({'loss': loss_k,
                         'acc': acc_k,
                         })


def learning_curve(data: pd.DataFrame,
                   injector: Injector,
                   knowledge: Iterable[Formula],
                   population_size: int = 10,
                   seed: int = 0,
                   epochs: int = 100,
                   batch_size: int = 32,
                   stop: EarlyStopping = None
                   ) -> pd.DataFrame:
    train_size, loss_knowledge, acc_knowledge, loss_vanilla, acc_vanilla = [], [], [], [], []
    train, test = train_test_split(data, test_size=798, random_state=seed, stratify=data.iloc[:, -1])
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]

    for i in range(1, 5):
        size = i*500
        sub_train, _ = train_test_split(train, train_size=size, random_state=seed, stratify=train.iloc[:, -1])
        train_x, train_y = sub_train.iloc[:, :-1], sub_train.iloc[:, -1:]

        tmp_loss_knowledge, tmp_acc_knowledge, tmp_loss_vanilla, tmp_acc_vanilla = [], [], [], []
        for j in range(0, population_size):
            seed = seed + i
            set_seed(seed)

            predictor_knowledge = injector.inject(knowledge)
            predictor_knowledge.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            predictor_knowledge.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=stop)
            loss, acc = predictor_knowledge.evaluate(test_x, test_y)
            tmp_loss_knowledge.append(loss)
            tmp_acc_knowledge.append(acc)
            del predictor_knowledge

            predictor_vanilla = clone_model(injector.predictor)
            predictor_vanilla.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            predictor_vanilla.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=stop)
            loss, acc = predictor_vanilla.evaluate(test_x, test_y)
            tmp_loss_vanilla.append(loss)
            tmp_acc_vanilla.append(acc)

        train_size.append(size)
        loss_knowledge.append(sum(tmp_loss_knowledge)/len(tmp_loss_knowledge))
        acc_knowledge.append(sum(tmp_acc_knowledge)/len(tmp_acc_knowledge))
        loss_vanilla.append(sum(tmp_loss_vanilla)/len(tmp_loss_vanilla))
        acc_vanilla.append(sum(tmp_acc_vanilla)/len(tmp_acc_vanilla))
        print('\n\n')
        print(train_size[-1], acc_knowledge[-1], acc_vanilla[-1])
        print('\n\n')

    return pd.DataFrame({'train_size': train_size,
                         'loss_knowledge': loss_knowledge,
                         'acc_knowledge': acc_knowledge,
                         'loss_vanilla': loss_vanilla,
                         'acc_vanilla': acc_vanilla,
                         })


def k_fold_cross_validation(data: pd.DataFrame,
                            injector: Injector,
                            knowledge: list[Formula],
                            use_knowledge: bool = True,
                            k: int = 10,
                            population_size: int = 30,
                            seed: int = 0,
                            epochs: int = 100,
                            batch_size: int = 32,
                            stop: EarlyStopping = None
                            ) -> pd.DataFrame:

    losses = []
    accuracies = []
    n, ie, ei = [], [], []

    for i in range(0, population_size):
        set_seed(seed + i)
        trained_predictors = []

        train, test = train_test_split(data, train_size=1000, random_state=seed, stratify=data.iloc[:, -1])
        test_n = test.loc[data[240] == 2]
        test_ie = test.loc[data[240] == 1]
        test_ei = test.loc[data[240] == 0]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        for train_indices, test_indices in k_fold.split(train.iloc[:, :-1], train.iloc[:, -1:]):
            predictor = injector.inject(knowledge) if use_knowledge else clone_model(injector.predictor)
            predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            train_x, train_y = train.iloc[train_indices, :-1], train.iloc[train_indices, -1:]
            # val_x, val_y = train.iloc[test_indices, :-1], train.iloc[test_indices, -1:]

            predictor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=stop)
            trained_predictors.append(predictor)

        predictions = np.array([predictor.predict(test_x) for predictor in trained_predictors])
        n_predictions = np.array([predictor.predict(test_n.iloc[:, :-1]) for predictor in trained_predictors])
        ie_predictions = np.array([predictor.predict(test_ie.iloc[:, :-1]) for predictor in trained_predictors])
        ei_predictions = np.array([predictor.predict(test_ei.iloc[:, :-1]) for predictor in trained_predictors])
        predictions = np.sum(predictions, axis=0) / k
        n_predictions = np.sum(n_predictions, axis=0) / k
        ie_predictions = np.sum(ie_predictions, axis=0) / k
        ei_predictions = np.sum(ei_predictions, axis=0) / k

        losses.append(np.mean(sparse_categorical_crossentropy(test_y, predictions)))
        accuracies.append(np.mean(sparse_categorical_accuracy(test_y, predictions)))

        n.append(np.mean(sparse_categorical_accuracy(test_n.iloc[:, -1:], n_predictions)))
        ie.append(np.mean(sparse_categorical_accuracy(test_ie.iloc[:, -1:], ie_predictions)))
        ei.append(np.mean(sparse_categorical_accuracy(test_ei.iloc[:, -1:], ei_predictions)))

        print(i+1)
        print(losses[-1])
        print(accuracies[-1])

        del trained_predictors

    return pd.DataFrame({'loss': losses, 'acc': accuracies, 'n_acc': n, 'ei_acc': ei, 'ie_acc': ie})


def leave_one_out(data: pd.DataFrame,
                  injector: Injector,
                  knowledge: list[Formula],
                  use_knowledge: bool = True,
                  seed: int = 0,
                  epochs: int = 100,
                  batch_size: int = 32,
                  stop: EarlyStopping = None
                  ) -> pd.DataFrame:
    losses = []
    accuracies = []
    set_seed(seed)

    for i in range(0, data.shape[0]):
        j = i+1 if i < data.shape[0] else i-1

        indices = sorted(set(range(0, i)).union(set(range(i + 1, data.shape[0]))))
        train, test = data.iloc[indices, :], data.iloc[i:j, :] if i < j else data.iloc[i, :]
        train = train.sample(frac=1, random_state=seed+i)
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]

        if use_knowledge:
            predictor = injector.inject(knowledge)
        else:
            predictor = clone_model(injector.predictor)
            predictor.set_weights(injector.predictor.get_weights())
        predictor.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

        predictor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=stop)

        loss, acc = predictor.evaluate(test_x, test_y, verbose=0)
        losses.append(loss)
        accuracies.append(acc)

        print(i + 1)
        print(sum(accuracies)/len(accuracies))

    return pd.DataFrame({'loss': [np.mean(losses)], 'acc': [np.mean(accuracies)]})

