import numpy as np
import pandas as pd

''' Question 1 '''
#print(np.__version__)

''' Question 2 '''
#print(pd.__version__)

df = pd.read_csv('data.csv')

'''
Question 3
Average price of BMW
'''
#print( df[ df['Make'] == 'BMW' ].MSRP.mean() )

'''
Question 4
Number of missing values in Engine HP after 2015
'''
# Summary
#res = df[ df['Year'] >= 2015 ].isnull().sum()
# Engine HP only
#res = df[ df['Year'] >= 2015 ]['Engine HP'].isnull().sum()

''' Question 5 '''
# 1: Calculate average Engine HP
hpavg = df['Engine HP'].mean()
# 2: fill Engine HP missing values with the calculated average
df['Engine HP'] = df['Engine HP'].fillna(hpavg)
# 3: Calculate average of HP again
new_hpavg = df['Engine HP'].mean()
#res = f'{round(hpavg)}, {round(new_hpavg)}'
#res = round(hpavg) == round(new_hpavg)

''' Question 6 '''
# 1: Select all the Rolls-Royce cars
res = df [ df['Make'] == 'Rolls-Royce' ]
# 2: Select only the specified 3 columns
res = res[['Engine HP', 'Engine Cylinders', 'highway MPG']]
# 3: Drop all duplicates
res = res.drop_duplicates()
# 4: Get the underlying NumPy array
X = res.values
# 5: Compute matrix-matrix multiplication between the transpose of X and X
XTX = X.T @ X
# 6: Invert XTX
invXTX = np.linalg.inv(XTX)
# 7: Sum of all the elements of the result
res = invXTX.sum()

''' Question 7 '''
# 1: Create the specified array
y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
# 2: Multiply the inverse of XTX with X.T and then multiply the result by y
w = invXTX @ X.T @ y
# 3: print first element of w
res = w[0]


print(res)