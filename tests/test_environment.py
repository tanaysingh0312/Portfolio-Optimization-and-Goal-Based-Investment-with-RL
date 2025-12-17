import unittest
import pandas as pd
from src.environment import Environment

class TestEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        env = Environment(stock_market_history=pd.read_csv('tests/close.csv'),
                          initial_cash_in_bank=10000,
                          buy_cost=0.1,
                          sell_cost=0.1,
                          limit_n_stocks=200)
        
        env.reset()

    def tearDown(self):
        pass
    
    def test_step(self):
        pass
    
    
if __name__ == '__main__':
    unittest.main()
