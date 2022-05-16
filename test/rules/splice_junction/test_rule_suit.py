import unittest
from resources.rules import get_splice_junction_rules, get_binary_datalog_rules, get_splice_junction_datalog_rules


class TestRuleSuit(unittest.TestCase):
    rules = get_splice_junction_rules('kb')
    ei_rule = rules[19]
    m_of_n = rules[18]
    datalog_ei_rule = get_splice_junction_datalog_rules([ei_rule])[0]
    datalog_binary_ei_rule = get_binary_datalog_rules([datalog_ei_rule])[0]
    datalog_m_of_n_rule = get_splice_junction_datalog_rules([m_of_n])[0]
    datalog_binary_m_of_n_rule = get_binary_datalog_rules([datalog_m_of_n_rule])[0]

    def test_get_datalog_rules(self):
        expected_rule = 'class(ei) ⇐ ¬(¬(X_3 = a) ∧ ¬(X_3 = c)) ∧ X_2 = a ∧ X_1 = g ∧ X1 = g ∧ X2 = t ∧ ¬(¬(X3 = a) ∧ '\
                        '¬(X3 = g)) ∧ X4 = a ∧ X5 = g ∧ X6 = t ∧ ¬(ei_stop())'
        self.assertEqual(self.datalog_ei_rule, expected_rule)

    def test_get_binary_datalog_rules(self):
        expected_rule = 'class(ei) ⇐ ¬(¬(X_3a = 1) ∧ ¬(X_3c = 1)) ∧ X_2a = 1 ∧ X_1g = 1 ∧ X1g = 1 ∧ X2t = 1 ∧ ¬(¬(X3a '\
                        '= 1) ∧ ¬(X3g = 1)) ∧ X4a = 1 ∧ X5g = 1 ∧ X6t = 1 ∧ ¬(ei_stop())'
        self.assertEqual(self.datalog_binary_ei_rule, expected_rule)

    def test_get_datalog_rules_m_of_n(self):
        expected_rule = 'pyramidine_rich() ⇐ 6 ≤ ((¬(¬(X_15 = c) ∧ ¬(X_15 = t))) + (¬(¬(X_14 = c) ∧ ¬(X_14 = t))) + (' \
                        '¬(¬(X_13 = c) ∧ ¬(X_13 = t))) + (¬(¬(X_12 = c) ∧ ¬(X_12 = t))) + (¬(¬(X_11 = c) ∧ ¬(X_11 = ' \
                        't))) + (¬(¬(X_10 = c) ∧ ¬(X_10 = t))) + (¬(¬(X_9 = c) ∧ ¬(X_9 = t))) + (¬(¬(X_8 = c) ∧ ¬(X_8 '\
                        '= t))) + (¬(¬(X_7 = c) ∧ ¬(X_7 = t))) + (¬(¬(X_6 = c) ∧ ¬(X_6 = t))))'
        self.assertEqual(self.datalog_m_of_n_rule, expected_rule)

    def test_get_binary_datalog_rules_m_of_n(self):
        expected_rule = 'pyramidine_rich() ⇐ 6 ≤ ((¬(¬(X_15c = 1) ∧ ¬(X_15t = 1))) + (¬(¬(X_14c = 1) ∧ ¬(X_14t = 1))) '\
                        '+ (¬(¬(X_13c = 1) ∧ ¬(X_13t = 1))) + (¬(¬(X_12c = 1) ∧ ¬(X_12t = 1))) + (¬(¬(X_11c = 1) ∧ ¬(' \
                        'X_11t = 1))) + (¬(¬(X_10c = 1) ∧ ¬(X_10t = 1))) + (¬(¬(X_9c = 1) ∧ ¬(X_9t = 1))) + (¬(¬(X_8c '\
                        '= 1) ∧ ¬(X_8t = 1))) + (¬(¬(X_7c = 1) ∧ ¬(X_7t = 1))) + (¬(¬(X_6c = 1) ∧ ¬(X_6t = 1))))'
        self.assertEqual(self.datalog_binary_m_of_n_rule, expected_rule)


if __name__ == '__main__':
    unittest.main()
