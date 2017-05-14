import unittest
import pandas as pd
import scipy.io as sio

class KaggleStockTests(unittest.TestCase):

    def __init__(self):
        super().__init__()
        self.data_pd = pd.read_csv('./data_raw/prices-split-adjusted.csv')
        self.data_mat = sio.loadmat('./daily_data.mat')['data']

    def test

def main():
    unittest.main()

if __name__ == '__main__':
    main()