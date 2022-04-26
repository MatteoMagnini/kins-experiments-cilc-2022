from typing import Iterable

import numpy as np
import pandas as pd
from psyki.ski import Injector, Formula
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import sparse_categorical_crossentropy, sparse_categorical_accuracy
from tensorflow.keras.models import clone_model


def run_experiments(data: pd.DataFrame,
                    injector: Injector,
                    knowledge: Iterable[Formula],
                    population_size: int = 30,
                    seed: int = 0,
                    epochs: int = 100,
                    batch_size: int = 32,
                    stop: EarlyStopping = None
                    ) -> pd.DataFrame:

    loss_i = []
    loss_k = []
    acc_i = []
    acc_k = []

    for i in range(0, population_size):
        seed = seed + i
        set_seed(seed)

        #predictor_i = clone_model(injector.predictor)
        predictor_k = injector.inject(knowledge)

        #predictor_i.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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

        #predictor_i.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs,
        #                batch_size=batch_size, verbose=0, callbacks=stop)
        #l_i, a_i = predictor_i.evaluate(test_x, test_y)

        #loss_i.append(l_i)
        #acc_i.append(a_i)

        del predictor_k#, predictor_i

    return pd.DataFrame({#'loss_no_knowledge': loss_i,
                         #'accuracy_no_knowledge': acc_i,
                         'loss': loss_k,
                         'acc': acc_k,
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

    for i in range(0, population_size):
        set_seed(seed + i)
        trained_predictors = []

        train, test = train_test_split(data, train_size=1000, random_state=seed, stratify=data.iloc[:, -1])
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        for train_indices, test_indices in k_fold.split(train.iloc[:, :-1], train.iloc[:, -1:]):
            predictor = injector.inject(knowledge) if use_knowledge else clone_model(injector.predictor)
            predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            train_x, train_y = train.iloc[train_indices, :-1], train.iloc[train_indices, -1:]
            val_x, val_y = train.iloc[test_indices, :-1], train.iloc[test_indices, -1:]

            predictor.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                          verbose=0, callbacks=stop)
            trained_predictors.append(predictor)

        predictions = np.array([predictor.predict(test_x) for predictor in trained_predictors])
        predictions = np.sum(predictions, axis=0)
        predictions = predictions / k

        losses.append(np.mean(sparse_categorical_crossentropy(test_y, predictions)))
        accuracies.append(np.mean(sparse_categorical_accuracy(test_y, predictions)))

        print(i+1)
        print(losses[-1])
        print(accuracies[-1])

        del trained_predictors

    return pd.DataFrame({'loss': losses, 'acc': accuracies})


