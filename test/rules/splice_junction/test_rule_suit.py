import unittest

from resources.rules import get_splice_junction_rules, get_binary_datalog_rules
from resources.rules.splice_junction import *


class TestRuleSuit(unittest.TestCase):
    rules = get_splice_junction_rules('kb')
    ei_rule = rules[19]
    print('Original rule :', ei_rule)
    datalog_ei_rule = get_datalog_rules([ei_rule])[0]
    datalog_binary_ei_rule = get_binary_datalog_rules([datalog_ei_rule])[0]

    def test_get_datalog_rules(self):
        expected_rule = 'class(ei) ← ¬(ei_stop()) ∧ (¬(X_3 = g) ∧ ¬(X_3 = t)) ∧ X_2 = a ∧ X_1 = g ∧ X1 = g ∧ X2 = t ∧ '\
                        '(¬(X3 = c) ∧ ¬(X3 = t)) ∧ X4 = a ∧ X5 = g ∧ X6 = t'
        self.assertEqual(self.datalog_ei_rule, expected_rule)

    def test_get_binary_datalog_rules(self):
        expected_rule = 'class(ei) ← ¬(ei_stop()) ∧ (¬(X_3g = 1) ∧ ¬(X_3t = 1)) ∧ X_2a = 1 ∧ X_1g = 1 ∧ X1g = 1 ∧ X2t '\
                        '= 1 ∧ (¬(X3c = 1) ∧ ¬(X3t = 1)) ∧ X4a = 1 ∧ X5g = 1 ∧ X6t = 1'
        self.assertEqual(self.datalog_binary_ei_rule, expected_rule)


if __name__ == '__main__':
    unittest.main()
