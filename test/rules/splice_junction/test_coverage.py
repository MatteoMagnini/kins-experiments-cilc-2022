import unittest
from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string
from psyki.ski.injectors import NetworkComposer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from resources.data import get_splice_junction_data, data_to_int, get_binary_data, \
    get_splice_junction_extended_feature_mapping
from resources.data.splice_junction import CLASS_MAPPING, AGGREGATE_FEATURE_MAPPING
from resources.rules import get_splice_junction_rules, get_splice_junction_datalog_rules, get_binary_datalog_rules


class TestCoverage(unittest.TestCase):

    rules = get_splice_junction_rules('kb')
    rules = get_splice_junction_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)
    data = get_splice_junction_data('data')
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    data = x.join(y)
    rules = [get_formula_from_string(rule) for rule in rules]

    inputs = Input((240,))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(4, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    predictor = Model(inputs, x)
    injector = NetworkComposer(predictor, get_splice_junction_extended_feature_mapping(), layer=2)

    def test_rule_coverage(self):
        predictor = self.injector.inject(self.rules)
        predict_ie = Model(predictor.inputs, predictor.layers[-3].output)
        results_ie = predict_ie.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        predict_ei = Model(predictor.inputs, predictor.layers[-4].output)
        result_ei = predict_ei.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        result_n = (~ result_ei) & (~ results_ie)

        self.assertTrue(sum(results_ie & (self.data.iloc[:, -1] == CLASS_MAPPING['ie'])), 295)
        self.assertTrue(sum(result_ei & (self.data.iloc[:, -1] == CLASS_MAPPING['ei'])), 31)
        self.assertTrue(sum(result_n & (self.data.iloc[:, -1] == CLASS_MAPPING['n'])), 1652)


if __name__ == '__main__':
    unittest.main()
