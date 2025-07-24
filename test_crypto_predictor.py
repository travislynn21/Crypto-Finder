import unittest
from unittest.mock import patch
import pandas as pd
from crypto_predictor import assess_portfolio, analyze_data, fetch_crypto_data

class TestCryptoPredictor(unittest.TestCase):

    @patch('builtins.input', side_effect=['Bitcoin', '1', 'Ethereum', '10', 'done'])
    def test_get_user_portfolio(self, mock_input):
        from crypto_predictor import get_user_portfolio
        portfolio = get_user_portfolio()
        self.assertEqual(portfolio, {'Bitcoin': 1.0, 'Ethereum': 10.0})

    def test_assess_portfolio(self):
        portfolio = {'Bitcoin': 1.0, 'Ethereum': 10.0}
        all_crypto_data = [
            {'name': 'Bitcoin', 'current_price': 50000},
            {'name': 'Ethereum', 'current_price': 4000}
        ]
        with patch('builtins.print') as mock_print:
            assess_portfolio(portfolio, all_crypto_data)
            mock_print.assert_any_call("\nAssessing Bitcoin:")
            mock_print.assert_any_call("  Amount: 1.0")
            mock_print.assert_any_call("  Current Price: $50,000.00")
            mock_print.assert_any_call("  Value: $50,000.00")
            mock_print.assert_any_call("\nAssessing Ethereum:")
            mock_print.assert_any_call("  Amount: 10.0")
            mock_print.assert_any_call("  Current Price: $4,000.00")
            mock_print.assert_any_call("  Value: $40,000.00")
            mock_print.assert_any_call("\nTotal Portfolio Value: $90,000.00")

    @patch('crypto_predictor.requests.get')
    def test_fetch_crypto_data(self, mock_get):
        mock_get.return_value.json.return_value = [
            {'name': 'Bitcoin', 'current_price': 50000, 'market_cap': 1000000000000, 'total_volume': 50000000000, 'price_change_percentage_24h': 5},
            {'name': 'Ethereum', 'current_price': 4000, 'market_cap': 500000000000, 'total_volume': 30000000000, 'price_change_percentage_24h': 3}
        ]
        data = fetch_crypto_data()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

if __name__ == '__main__':
    unittest.main()
