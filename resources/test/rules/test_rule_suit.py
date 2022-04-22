import unittest
from resources.rules import get_rules, get_datalog_rules, get_rules_with_explicit_variables, get_rules_with_int


class TestRuleSuit(unittest.TestCase):
    rules = get_rules('kb')
    ei_rule = rules[19]
    datalog_ei_rule = get_datalog_rules([ei_rule])[0]
    datalog_ei_rule_explicit_variable = get_rules_with_explicit_variables([datalog_ei_rule])[0]
    valuer_mapping = {'ei': 0, 'ie': 1, 'n': 2, 'a': 3, 'c': 4, 'g': 5, 't': 6, 'd': 7, 's': 8, 'r': 9}
    datalog_ei_rule_with_int = get_rules_with_int([datalog_ei_rule_explicit_variable], valuer_mapping)[0]

    def test_get_datalog_rules(self):
        expected_rule = 'class(ei) ← ¬(ei_stop()) ∧ (¬(X_3 = g) ∧ ¬(X_3 = t)) ∧ X_2 = a ∧ X_1 = g ∧ X1 = g ∧ X1 = t ∧ '\
                        '(¬(X2 = c) ∧ ¬(X2 = t)) ∧ X3 = a ∧ X4 = g ∧ X5 = t'
        self.assertEqual(self.datalog_ei_rule, expected_rule)

    def test_get_rules_with_explicit_variables(self):
        expected_rule = 'class(ei) ← ¬(ei_stop()) ∧ (¬(((X_3 = g) ∨ (X_3 = d) ∨ (X_3 = n) ∨ (X_3 = s) ∨ (X_3 = r))) ∧ '\
                        '¬(((X_3 = t) ∨ (X_3 = d) ∨ (X_3 = n)))) ∧ ((X_2 = a) ∨ (X_2 = d) ∨ (X_2 = n) ∨ (X_2 = r)) ∧ ' \
                        '((X_1 = g) ∨ (X_1 = d) ∨ (X_1 = n) ∨ (X_1 = s) ∨ (X_1 = r)) ∧ ((X1 = g) ∨ (X1 = d) ∨ (X1 = ' \
                        'n) ∨ (X1 = s) ∨ (X1 = r)) ∧ ((X1 = t) ∨ (X1 = d) ∨ (X1 = n)) ∧ (¬(((X2 = c) ∨ (X2 = n) ∨ (X2 '\
                        '= s))) ∧ ¬(((X2 = t) ∨ (X2 = d) ∨ (X2 = n)))) ∧ ((X3 = a) ∨ (X3 = d) ∨ (X3 = n) ∨ (X3 = r)) ' \
                        '∧ ((X4 = g) ∨ (X4 = d) ∨ (X4 = n) ∨ (X4 = s) ∨ (X4 = r)) ∧ ((X5 = t) ∨ (X5 = d) ∨ (X5 = n))'
        self.assertEqual(self.datalog_ei_rule_explicit_variable, expected_rule)

    def test_get_rules_with_int(self):
        expected_rule = 'class(ei) ← ¬(ei_stop()) ∧ (¬(((X_3 = 5) ∨ (X_3 = 7) ∨ (X_3 = 2) ∨ (X_3 = 8) ∨ (X_3 = 9))) ∧ '\
                        '¬(((X_3 = 6) ∨ (X_3 = 7) ∨ (X_3 = 2)))) ∧ ((X_2 = 3) ∨ (X_2 = 7) ∨ (X_2 = 2) ∨ (X_2 = 9)) ∧ ' \
                        '((X_1 = 5) ∨ (X_1 = 7) ∨ (X_1 = 2) ∨ (X_1 = 8) ∨ (X_1 = 9)) ∧ ((X1 = 5) ∨ (X1 = 7) ∨ (X1 = ' \
                        '2) ∨ (X1 = 8) ∨ (X1 = 9)) ∧ ((X1 = 6) ∨ (X1 = 7) ∨ (X1 = 2)) ∧ (¬(((X2 = 4) ∨ (X2 = 2) ∨ (X2 '\
                        '= 8))) ∧ ¬(((X2 = 6) ∨ (X2 = 7) ∨ (X2 = 2)))) ∧ ((X3 = 3) ∨ (X3 = 7) ∨ (X3 = 2) ∨ (X3 = 9)) ' \
                        '∧ ((X4 = 5) ∨ (X4 = 7) ∨ (X4 = 2) ∨ (X4 = 8) ∨ (X4 = 9)) ∧ ((X5 = 6) ∨ (X5 = 7) ∨ (X5 = 2))'
        print(self.datalog_ei_rule_explicit_variable)
        self.assertEqual(self.datalog_ei_rule_with_int, expected_rule)


if __name__ == '__main__':
    unittest.main()
