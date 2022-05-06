from resources.results import get_results
import numpy as np

data = get_results('sj-results-k-fold-cross-val-no-rules')
print(100*(1-np.mean(data['n_acc'])))
print(100*(1-np.mean(data['ei_acc'])))
print(100*(1-np.mean(data['ie_acc'])))
