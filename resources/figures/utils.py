import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from resources.results import get_results


def learning_curve(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data['train_size'], 1 - data['acc_knowledge'], linewidth=2.0)
    ax.plot(data['train_size'], 1 - data['acc_vanilla'], linewidth=2.0)

    ax.set(xlim=(500, 2000),
           ylim=(0.05, 0.1))
    # fig.show()
    fig.savefig('learning-curve-500-2000.pdf')


def sj_class_error_rate_standings():

    data = pd.DataFrame({
        'KINS': [3.52, 7.21, 7.15],
        'DNN': [3.46, 7.27, 8.33],
        'KBANN': [4.26, 7.56, 8.47],
        'Backprop': [5.29, 5.74, 10.75],
        'PEBLS': [6.86, 8.18, 7.55],
        # 'Perceptron': [3.99, 16.32, 17.41],
        'ID3': [8.84, 10.58, 13.99],
        'NearestNeighbour': [31.11, 11.65, 9.09],
    })

    plt.rcdefaults()
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    axs[0].barh(np.arange(data.shape[1]), data.iloc[0, :], color=['red'] + (data.shape[1]-1)*['#1f77b4'])
    axs[0].set_yticks(np.arange(data.shape[1]), labels=data.columns)
    axs[0].invert_yaxis()  # labels read top-to-bottom
    axs[0].set_title('Nothing')

    axs[1].barh(np.arange(data.shape[1]), data.iloc[1, :], color=['red'] + (data.shape[1]-1)*['#1f77b4'])
    axs[1].set_yticks(np.arange(data.shape[1]), labels=data.columns)
    axs[1].invert_yaxis()  # labels read top-to-bottom
    axs[1].set_title('EI')

    axs[2].barh(np.arange(data.shape[1]), data.iloc[2, :], color=['red'] + (data.shape[1]-1)*['#1f77b4'])
    axs[2].set_yticks(np.arange(data.shape[1]), labels=data.columns)
    axs[2].invert_yaxis()  # labels read top-to-bottom
    axs[2].set_title('IE')

    fig.supxlabel('Error rate')
    fig.savefig('error-rate.pdf', transparent=True)


data = get_results('sj-results-curve-500-2000')
learning_curve(data)

sj_class_error_rate_standings()


