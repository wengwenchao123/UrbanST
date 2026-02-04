import csv
import os
import pickle

import numpy as np

# f = open('adj_mx.pkl', 'rb')
with open('adj_mx.pkl', 'rb') as f:
    adj_mx = pickle.load(f)
# adj_mx = pickle.load(f)
print(adj_mx)