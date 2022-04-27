import unittest

from resources.rules import get_splice_junction_rules, get_binary_datalog_rules, get_splice_junction_datalog_rules


class TestRuleSuit(unittest.TestCase):
    rules = get_splice_junction_rules('kb')
    ei_rule = rules[19]
    m_of_n = rules[18]
    print('Original EI rule :', ei_rule)
    datalog_ei_rule = get_splice_junction_datalog_rules([ei_rule])[0]
    datalog_binary_ei_rule = get_binary_datalog_rules([datalog_ei_rule])[0]
    print('\nOriginal pyramidine-rich rule :', m_of_n)
    datalog_m_of_n_rule = get_splice_junction_datalog_rules([m_of_n])[0]
    print('Datalog pyramidine-rich_rule rule :', datalog_m_of_n_rule)
    datalog_binary_m_of_n_rule = get_binary_datalog_rules([datalog_m_of_n_rule])[0]
    print('Final Datalog pyramidine-rich_rule rule :', datalog_binary_m_of_n_rule)

    def test_get_datalog_rules(self):
        expected_rule = 'class(ei) ← ¬(ei_stop()) ∧ (¬(X_3 = g) ∧ ¬(X_3 = t)) ∧ X_2 = a ∧ X_1 = g ∧ X1 = g ∧ X2 = t ∧ '\
                        '(¬(X3 = c) ∧ ¬(X3 = t)) ∧ X4 = a ∧ X5 = g ∧ X6 = t'
        self.assertEqual(self.datalog_ei_rule, expected_rule)

    def test_get_binary_datalog_rules(self):
        expected_rule = 'class(ei) ← ¬(ei_stop()) ∧ (¬(X_3g = 1) ∧ ¬(X_3t = 1)) ∧ X_2a = 1 ∧ X_1g = 1 ∧ X1g = 1 ∧ X2t '\
                        '= 1 ∧ (¬(X3c = 1) ∧ ¬(X3t = 1)) ∧ X4a = 1 ∧ X5g = 1 ∧ X6t = 1'
        self.assertEqual(self.datalog_binary_ei_rule, expected_rule)

    def test_get_datalog_rules_m_of_n(self):
        expected_rule = 'pyramidine_rich() ← 6 ≤ (((¬(X_15 = a) ∧ ¬(X_15 = g))) + ((¬(X_14 = a) ∧ ¬(X_14 = g))) + ((' \
                        '¬(X_13 = a) ∧ ¬(X_13 = g))) + ((¬(X_12 = a) ∧ ¬(X_12 = g))) + ((¬(X_11 = a) ∧ ¬(X_11 = g))) ' \
                        '+ ((¬(X_10 = a) ∧ ¬(X_10 = g))) + ((¬(X_9 = a) ∧ ¬(X_9 = g))) + ((¬(X_8 = a) ∧ ¬(X_8 = g))) ' \
                        '+ ((¬(X_7 = a) ∧ ¬(X_7 = g))) + ((¬(X_6 = a) ∧ ¬(X_6 = g))))'
        self.assertEqual(self.datalog_m_of_n_rule, expected_rule)

    def test_get_binary_datalog_rules_m_of_n(self):
        expected_rule = 'pyramidine_rich() ← 6 ≤ (((¬(X_15a = 1) ∧ ¬(X_15g = 1))) + ((¬(X_14a = 1) ∧ ¬(X_14g = 1))) + '\
                        '((¬(X_13a = 1) ∧ ¬(X_13g = 1))) + ((¬(X_12a = 1) ∧ ¬(X_12g = 1))) + ((¬(X_11a = 1) ∧ ¬(X_11g '\
                        '= 1))) + ((¬(X_10a = 1) ∧ ¬(X_10g = 1))) + ((¬(X_9a = 1) ∧ ¬(X_9g = 1))) + ((¬(X_8a = 1) ∧ ' \
                        '¬(X_8g = 1))) + ((¬(X_7a = 1) ∧ ¬(X_7g = 1))) + ((¬(X_6a = 1) ∧ ¬(X_6g = 1))))'
        self.assertEqual(self.datalog_binary_m_of_n_rule, expected_rule)


if __name__ == '__main__':
    unittest.main()
