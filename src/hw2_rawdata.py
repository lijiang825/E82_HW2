#!/usr/bin/env python

# Very basic exploratory stuff
import seaborn as sns
import matplotlib.pyplot as plt

# import raw data
from hw2_config import *
from hw2_data import dat

## -------------------------------------------------------------------
### Simple Exploratory Stuff

def data_info(data):
    print(f"Columns: {data.columns}")
    print(f"Shape: {data.shape}")
    print(f"Data:\n{data.head(5)}")
    print(f"...")
    print(f"{data.tail(5)}")

data_info(dat)

# Papers over time: generally exponentially growing #papers/year
def data_papers_per_year(data):
    sns.countplot(y='year', data=data)
    plt.grid(axis='x')
    plt.title("NIPS Articles per Year")
    plt.xlabel("Articles")
    plt.ylabel("Year")
    plt.show()

data_papers_per_year(dat)
