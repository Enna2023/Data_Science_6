
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno

train = pd.read_csv('D:/Resume/fiveweek/week1/eda/train_houseprices.csv')
test = pd.read_csv('D:/Resume/fiveweek/week1/eda/test_houseprices.csv')

train.describe()
train.head()
train.tail()
train.shape , test.shape

# examine numerical features in the train dataset

train_num = train.select_dtypes(include=[np.number])
train_num.head()
train_num.columns

# examine categorical features in the train dataset

train_cat = train.select_dtypes(include=[np.object])
train_cat.head()
train_cat.columns

# Visualising missing values for a sample of 250
msno.matrix(train.sample(250))
msno.matrix(train.sample(300))

# Heatmap
'''
The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another:'''

msno.heatmap(train)
msno.bar(train.sample(1000))

# Dendrogram
'''The dendrogram allows you to more fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap:'''

msno.dendrogram(train)
train.skew()
train.kurt()

y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)

'''It is apparent that SalePrice doesn't follow normal distribution, so before performing regression it has to be transformed. While log transformation does pretty good job, best fit is unbounded Johnson distribution.'''

sns.distplot(train.skew(),color='blue',axlabel ='Skewness')

plt.figure(figsize = (12,8))
sns.distplot(train.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)
#plt.hist(train.kurt(),orientation = 'vertical',histtype = 'bar',label ='Kurtosis', color ='blue')
plt.show()

plt.hist(train['SalePrice'],orientation = 'vertical',histtype = 'bar', color ='blue')
plt.show()

target = np.log(train['SalePrice'])
target.skew()
plt.hist(target,color='blue')
plt.show()

# Finding Correlation coefficients between numeric features and SalePrice
correlation = train_num.corr()
print(correlation['SalePrice'].sort_values(ascending = False),'\n')



