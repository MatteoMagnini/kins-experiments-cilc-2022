import unittest
from psyki.logic.datalog.grammar.adapters import Antlr4
from psyki.ski.injectors import NetworkComposer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from resources.data import get_splice_junction_data, data_to_int, get_binary_data, get_splice_junction_extended_feature_mapping
from resources.rules import get_splice_junction_rules, get_splice_junction_datalog_rules, get_binary_datalog_rules


FEATURES = ['a', 'c', 'g', 't']
FEATURE_MAPPING = {'a': ('a', ),
                   'c': ('c', ),
                   'g': ('g', ),
                   't': ('t', ),
                   'd': ('a', 'g', 't'),
                   'm': ('a', 'c'),
                   'n': ('a', 'c', 'g', 't'),
                   'r': ('a', 'g'),
                   's': ('c', 'g'),
                   'y': ('c', 't')}
CLASS_MAPPING = {'ei': 0,
                 'ie': 1,
                 'n': 2}


class TestCoverage(unittest.TestCase):

    rules = get_splice_junction_rules('kb')
    rules = get_splice_junction_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)

    data = get_splice_junction_data('data')
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(data.iloc[:, :-1], FEATURE_MAPPING)
    # x.columns = list(get_splice_junction_extended_feature_mapping(FEATURES).keys())
    y.columns = [x.shape[1]]
    data = x.join(y)

    ie = data.loc[data[240] == CLASS_MAPPING['ie']]
    ei = data.loc[data[240] == CLASS_MAPPING['ei']]

    rules = [Antlr4().get_formula_from_string(rule) for rule in rules]

    inputs = Input((240,))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(4, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    predictor = Model(inputs, x)
    injector = NetworkComposer(predictor, get_splice_junction_extended_feature_mapping(FEATURES))

    def test_rule_coverage(self):
        predictor = self.injector.inject(self.rules)

        predict_ie = Model(predictor.inputs, predictor.layers[-3].output)
        ie_true_positive = predict_ie.predict(self.ie.iloc[:, :-1])
        print('IE true positive: ' + str(sum(ie_true_positive)[0]/self.ie.shape[0]*100) + '% (' + str(self.ie.shape[0]) + ' total positive)')

        predict_ei = Model(predictor.inputs, predictor.layers[-4].output)
        ei_true_positive = predict_ei.predict(self.ei.iloc[:, :-1])
        print('EI true positive: ' + str(sum(ei_true_positive)[0]/self.ei.shape[0]*100) + '% (' + str(self.ei.shape[0]) + ' total positive)')

