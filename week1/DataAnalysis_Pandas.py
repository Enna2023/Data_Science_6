
import pandas as pd

pd.__version__

# 2.1. Pandas Series Object

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data) 
data.values
data.index
# Like with a NumPy array, data can be accessed by the associated index via the familiar Python square-bracket slicing notation:
data[1]
data[1:3]

# 2.1.1. Series as generalized NumPy array
data = pd.Series([0.25, 0.5, 0.75, 1.0],
index=['a', 'b', 'c', 'd'])
data
data['b']

'''
We can even use non-contiguous or non-sequential indices (index), considering the number of defined indices corresponds to the number of added values in the Series:'''

data = pd.Series([0.25, 0.5, 0.75, 1.0],
index=[2, 5, 3, 7])
data
data[5]

# 2.1.2 Series as specialized dictionaries
'''A dictionary is a structure that maps arbitrary keys to a set of arbitrary values, and a Series is a structure which maps typed keys to a set of typed values. We can take advantage of these similarities to create a Series from a dictionary, where the keys are the indices of the Series and the values are those associated with these indices.'''

population_dict = {'California': 38332521,
'Texas': 26448193,
'New York': 19651127,
'Florida': 19552860,
'Illinois': 12882135}
population = pd.Series(population_dict)
population
population['California']
population['California':'Florida']

# 2.2. Pandas DataFrame Object
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area
states = pd.DataFrame({'population': population,'area': area})
states
states.index
states.columns
'''There are some functions and attributes that allow us to observe basic information about the data stored in a DataFrame object:

DataFrame.head() -> returns the content of the first 5 rows, by default
DataFrame.tail() -> returns the content of the last 5 rows, by default
DataFrame.shape -> returns a tuple of the form (num_rows, num_columns)
DataFrame.columns -> returns the name of the columns
DataFrame.index -> returns the index of the rows'''

states.head(3) # The first three rows
states.tail(3) # The last three rows

# 2.3. Pandas Index Object
ind = pd.Index([2, 3, 5, 7, 11])
ind
# This index object can be sliced as a list or a numpy array:
ind[1] # Accessing element in position 1
ind[::2] # Accessing elements starting from position 0 through all the elements two by two.
print(' Size:',ind.size,'\n',
'Shape:',ind.shape,'\n',
'Dimension:',ind.ndim,'\n',
'Data type:',ind.dtype)
# One difference between Index objects and Numpy arrays is that indices are immutable.

# 3.1. Data Selection in Series

data = pd.Series([0.25, 0.5, 0.75, 1.0],
index=['a', 'b', 'c', 'd'])
data

data['d'] = 0.95 # updating the value associated with 'd' index in Series object
data

data['e'] = 1.25 # New value added with 'e' index in Series object
data

# slicing by implicit integer index
data[0:2]
# slicing by explicit index
data['a':'c']
# express indexing
data[['a', 'e']]

'''
Notice that when slicing with an explicit index (i.e., data ['a':'c']), the final index is included in the slice, while when slicing with an implicit index (i.e., data[0:2]), the final index is excluded from the slice. When we slicing through a list (i.e., data [['a', 'e']]), all indices are equally accessed.'''

# 3.4. Indexers: loc, iloc for DataFrame
# 3.4.1. loc attribute
# The loc attribute allows indexing and slicing that always references the explicit index (the explicit name of the index):
states
states.loc[:'Illinois', :'area']

# 3.4.2. iloc attribute
# The iloc attribute allows indexing and slicing that always references the implicit Python-style index:
states.iloc[:3, :2]
# In this example we are slicing the DataFrame from index 0 by default to 3 exlusive, and from column 0 by default to 2 exlusive

'''Total_Candidates = {'absolute_beginners': 785, 'beginners': 825, 'intermediat_advanced': 602} # this is true data
Active_Candidates = {'absolute_beginners': 500, 'beginners': 425, 'intermediat_advanced': 300}  # this is hypothetical data'''

# Create a Pandas DataFrame using above information (name your Dataframe as DPhi)
Total_Candidates = {'absolute_beginners': 785, 'beginners': 825, 'intermediat_advanced': 602}
Active_Candidates = {'absolute_beginners': 500, 'beginners': 425, 'intermediat_advanced': 300}

data = {'Total_Candidates': Total_Candidates, 'Active_Candidates': Active_Candidates}

DPhi = pd.DataFrame(data)
print(DPhi)

# get all the columns in DPhi
DPhi.columns

# Get the information of total candidates present in each batches using dictionary-style indexing
DPhi['Total_Candidates']

# Find the number of candidates for each batches who are not active and add this information to the dataframe DPhi.
DPhi['Not_Active_Candidates'] = DPhi['Total_Candidates'] - DPhi['Active_Candidates']
DPhi

# Find the percentage of active candidates in each batches and add this information to the dataframe DPhi.
DPhi['Percentage_Active_Candidates'] = (DPhi['Active_Candidates'] / DPhi['Total_Candidates']) * 100
DPhi

# Get all the batches where percentage of active candidates are greater than 60%
DPhi[DPhi['Percentage_Active_Candidates'] > 60]

# 3.5. Subsetting a Dataframe
DPhi[DPhi['Percentage_Active_Candidates'] > 60]

DPhi['Percentage_Active_Candidates'] > 60
DPhi[(DPhi['Percentage_Active_Candidates'] > 60) & (DPhi['Total_Candidates'] > 600)]
DPhi[(DPhi['Percentage_Active_Candidates'] > 60) | (DPhi['Total_Candidates'] > 600)] 
# | is used for or
DPhi[~((DPhi['Percentage_Active_Candidates'] < 60) & (DPhi['Total_Candidates'] > 600))] 

# 4. Data Wrangling

'''
1. Data structuring:

The first step in the data wrangling process is to separate the relevant data into multiple columns, so that the analysis can be run grouping by common values in a separate way. In turn, if there are columns that are not desired or that will not be relevant to the analysis, this is the phase to filter the data or mix together some of their columns.

2. Data Cleaning

In this step, the data is cleaned up for high-quality analysis. Null values are handled, and the data format is standardized. This step is also known as Data Preparation.

3. Data Enriching

After cleaning, the data is enriched by increasing some variables in what is known as Data Augmentation and using additional sources to enrich them for the following stages of processing.
'''

# 5. Handling Missing Data
'''
Missing completely at random (MCAR): when the fact that the data is missing is independent of the observed and unobserved data.
Missing at random (MAR): when the fact that the data is missing is systematically related to the observed but not the unobserved data.
Missing not at random (MNAR): when the missingness of data is related to events or factors which are not measured by the researcher.'''

# 5.1. NaN and None in Pandas
import numpy as np
import pandas as pd

pd.Series([1, np.nan, 2, None])

# 5.2. Operations on Missing Values
'''
There are several useful methods for detecting, removing, and replacing missing values in Pandas data structures:

isnull(): generates a boolean mask indicating missing values
notnull(): generates a boolean mask of non-missing values. Is the opposite of isnull().
dropna(): returns a filtered version of the data, without missing values.
fillna(): returns a copy of the data with missing values filled or imputed with a desired strategy.
'''

data = pd.Series([1, np.nan, 'hello', None])
data
data.isnull()
data.notnull()
data[data.notnull()]

# 5.3. Dropping missing values
data.dropna()
data

df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]])
df
# By default, dropna() will drop all rows in which any missing value is present:
df.dropna()
# Alternatively, you can drop missing values along a different axis; axis=1 drops all columns containing a missing value:
df.dropna(axis='columns')

'''
But this drops some good data as well; you might rather be interested in dropping rows or columns with all NaN values, or a majority of NaN values. This can be specified through the how or thresh parameters, which allow fine control of the number of nulls to allow through.

The default is how='any', such that any row or column (depending on the axis keyword) containing a null value will be dropped. You can also specify how='all', which will only drop rows/columns that are all null values:'''

df = pd.DataFrame([[1, np.nan, 2, np.nan],
[2, 3, 5, np.nan],
[np.nan, 4, 6, np.nan]])
df
df.dropna(axis='columns', how='all')


# 5.4. Filling missing values

'''
There are four types of treatment that can be given, in that order, to unwanted non-existent or missing data:

Treatment 1: Ignore the missing or unwanted data in some columns, considering that in other columns of the same rows there are important or relevant data for the study.
Treatment 2: Replace the missing or unwanted data with values that represent an indicator of nullity.
Treatment 3: Replace the missing, nonexistent or unwanted data with interpolated values that are related to the trend of the data that is present.
Treatment 4: Delete the missing data, with the certainty that valuable information will not be lost when analyzing the data.
You can apply Treatment 2 and Treatment 3 in-place using the isnull() method as a mask, but because it is such a common operation Pandas provides the fillna() method, which returns a copy of the array with the missing values replaced.
'''
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)
# forward-fill
data.fillna(method='ffill') # ffill is used to fill the missing values with the previous value in the series
# back-fill
data.fillna(method='bfill') # bfill is used to fill the missing values with the next value in the series

df
df.fillna(method='ffill', axis=1) # forward-fill along the columns
df.fillna(method='ffill', axis=0) # forward-fill along the rows

# 6. Pandas String Operations
data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]
# This is perhaps sufficient to work with some data, but it will break if there are any missing values

import pandas as pd
names = pd.Series(data)
names
names.str.capitalize()

# 6.1. String Methods
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
'Eric Idle', 'Terry Jones', 'Michael Palin'])
monte.str.lower() # Parse values to string and transform all characters to lowercase
monte.str.len() # Parse values to string and calculates their length
# Parse values to string and calculates a mask of string values starting by 'T'
monte.str.startswith('T')
monte.str.split() # Parse values to string and splits them by ' ' character, by default

country = ['Netherland', 'Germany', 'Peru', 'Israel', 'Madagascar']
year = [2002, 2002, 1957, 2007, 1967]
population = [16122830.0, np.nan, 9146100.0, 6426679.0, 6334556.0]
continent = ['Europe', 'europe', 'Americas', 'asia', 'Africa']
# Create a Dataframe object which contains all the lists values as Series. The final DataFrame should be named as country_info, containing 4 columns and 5 rows.
country_info = pd.DataFrame({'country': country, 'year': year, 'population': population, 'continent': continent})
country_info

# Delete the rows which contains missing values
country_info.dropna()

# Capitalize all the continents in continent column.
country_info['continent'] = country_info['continent'].str.capitalize()
country_info

# Get the length of each country's names.
country_info['country'].str.len()

# 7. Concatenate Series

ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2], axis=0)
pd.concat([ser1, ser2], axis=1)

country = ['Netherland', 'Germany', 'Peru', 'Israel', 'Madagascar']
gdp_per_cap = [33724.757780, 30035.801980, 4245.256698, 25523.277100, 1634.047282]

# Create a Dataframe object which contains all the lists values as Series. The final DataFrame should be named as country_info, containing 2 columns and 5 rows.
country_info = pd.DataFrame({'country': country, 'gdp_per_cap': gdp_per_cap})
country_info
# Concatenate the two dataframes: country_info and country_gdp with axis=0 and name it concat_data
country = pd.DataFrame(country)
gdp_per_cap = pd.DataFrame(gdp_per_cap)
concat_data = pd.concat([country, gdp_per_cap], axis=1)
concat_data = pd.concat([country, gdp_per_cap], axis=0)
concat_data
# Check if there are any null values in concat_data
concat_data.isnull()
# Find total numer of missing values in each column. hint: Use .isnull() and .sum() functions
concat_data.isnull().sum()






