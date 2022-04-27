import unittest
from psyki.logic.datalog.grammar.adapters import Antlr4
from psyki.ski.injectors import NetworkComposer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from resources.data import get_promoters_data, data_to_int, get_binary_data, get_promoter_extended_feature_mapping
from resources.rules import get_promoters_rules, get_promoters_datalog_rules, get_binary_datalog_rules


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
CLASS_MAPPING_PROMOTERS = {'-': 0,
                           '+': 1}


class TestCoverage(unittest.TestCase):
    rules = get_promoters_rules('kb')
    rules = get_promoters_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)

    data = get_promoters_data('data')
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING_PROMOTERS)
    x = get_binary_data(data.iloc[:, :-1], FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    data = x.join(y)

    rules = [Antlr4().get_formula_from_string(rule) for rule in rules]

    inputs = Input((228,))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(4, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    predictor = Model(inputs, x)
    injector = NetworkComposer(predictor, get_promoter_extended_feature_mapping(FEATURES))

    def test_rule_coverage(self):
        predictor = self.injector.inject(self.rules)
        # predictor.summary()
        predictor2 = Model(predictor.inputs, predictor.layers[-5].output)
        results = predictor2.predict(self.data.iloc[:, :-1])
        print(sum(results)/results.shape[0])
        predictor2 = Model(predictor.inputs, predictor.layers[-4].output)
        results = predictor2.predict(self.data.iloc[:, :-1])
        print(sum(results) / results.shape[0])

