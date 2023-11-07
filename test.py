import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


array = [1,2,3]

iter = combinations(array, 2)

for a,b in iter:
    print(a)
    print(b)
    print("----")