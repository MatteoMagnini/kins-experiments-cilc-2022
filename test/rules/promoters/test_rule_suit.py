import unittest
from resources.rules import get_promoters_rules, get_binary_datalog_rules, get_promoters_datalog_rules


class TestRuleSuit(unittest.TestCase):
    rules = get_promoters_rules('kb')
    minus35_rule = rules[0]
    print('Original rule :', minus35_rule)
    datalog_minus35_rule = get_promoters_datalog_rules([minus35_rule])[0]
    datalog_binary_minus_35_rule = get_binary_datalog_rules([datalog_minus35_rule])[0]

    def test_get_datalog_rules(self):
        expected_rule = 'minus_35() ← X_37 = c ∧ X_36 = t ∧ X_35 = t ∧ X_34 = g ∧ X_33 = a ∧ X_32 = c'
        self.assertEqual(self.datalog_minus35_rule, expected_rule)

    def test_get_binary_datalog_rules(self):
        expected_rule = 'minus_35() ← X_37c = 1 ∧ X_36t = 1 ∧ X_35t = 1 ∧ X_34g = 1 ∧ X_33a = 1 ∧ X_32c = 1'
        self.assertEqual(self.datalog_binary_minus_35_rule, expected_rule)


if __name__ == '__main__':
    unittest.main()
