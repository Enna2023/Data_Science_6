# https://www.kaggle.com/code/steffanj/titanic-preprocessing-eda-and-ml-in-python/notebook


'''
Preprocessing/cleaning of the provided data
Exploratory analysis of the data
Preprocessing for machine learning
Fitting machine learning models
Predicting test samples
'''
# Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Read in the data
train_data = pd.read_csv('D:/Resume/fiveweek/week1/DataPre_processing/titanic_data.csv')
train_data.head(5)

train_data.info()

# First of all, the DataFrame index does not correspond to the PassengerId column. Let's check if the values in PassengerId are unique using numpy.unique():
(np.unique(train_data['PassengerId'].values).size, 
np.unique(train_data['PassengerId'].values).size == train_data.shape[0])

# set the index to PassengerId
train_data.set_index(['PassengerId'], inplace=True)
train_data.head(0)

'''
extract the titles (Mr., Miss. etc.) of the passengers and use the title as a categorical/nominal feature. In the 'Name' column of our DataFrame, the titles are followed by a '.', which will help us extracting them. 
'''
import re
patt = re.compile('\s(\S+\.)') # \s matches any whitespace character, \S matches any non-whitespace character, + matches one or more of the preceding character, . matches any character except a newline

titles = np.array([re.search(patt, i)[1] for i in train_data['Name'].values]) # re.search() returns a match object, which we can use to extract the title

print('Unique titles ({}): \n{}'.format(np.unique(titles).shape[0], np.unique(titles))) # print the unique titles

print('Number of titles that are NaN/Null: {}'.format(pd.isnull(titles).sum()))

#  include the titles as a new feature 'Title' in the DataFrame, and drop the 'Name' feature:
train_data['Title'] = titles
train_data.drop('Name', axis=1, inplace=True)
train_data.head(5)

# count the number of occurrences for each title
train_data['Title'].value_counts()

'''
most of the titles occur very infrequently. Fitting our models to these titles might mean we would be overfitting. Let's group 'Mlle' and 'Mme' with their English counterparts 'Miss' and 'Mrs'. 'Ms' will be grouped with 'Miss'; 'Capt', 'Col' and 'Major' will be put in an 'Army' category, and 'Countess', 'Don', 'Jonkheer', 'Lady' and 'Sir' will be put in a 'Noble' category. 'Dr' will be kept as a category:
'''
train_data['Title'] = train_data['Title'].replace('Mlle.','Miss.')
train_data['Title'] = train_data['Title'].replace('Ms.','Miss.')  
train_data['Title'] = train_data['Title'].replace('Mme.','Mrs.')
train_data['Title'] = train_data['Title'].replace(['Capt.','Col.','Major.'],'Army.')
train_data['Title'] = train_data['Title'].replace(['Countess.','Don.','Jonkheer.','Lady.','Sir.'],'Noble.')
train_data['Title'].value_counts()
train_data[['Title','Survived']].groupby(train_data['Title']).mean() 

# drop the 'Ticket' column, this feature is not likely to add anything to the analysis.

train_data = train_data.drop('Ticket', axis=1)
train_data.head()

'''
Individual cabin codes are not likely to have much predictive power in our problem. However, the cabin codes can be split in categories based on the letter in the code, e.g. 'C' or 'D'. These letters might encode cabin class and thus social status, which might have predictive power on the survival odds. Since we don't know the ordering of the cabin categories, if there is any at all, this feature will be a nominal feature. Passengers without a class will have entries of 'None'.'''

def getCabinCat(cabin_code):
    if pd.isnull(cabin_code):
        cat = 'None' # Use a string so that it is clear that this is 
                     # a category on its own
    else:
        cat = cabin_code[0]
    return cat

train_data['CabinCat'] = train_data['Cabin'].apply(getCabinCat)
train_data = train_data.drop('Cabin', axis=1)
train_data.head()

# the distribution of passengers amongst the cabin categories
train_data['CabinCat'].value_counts()

# Exploratory Data Analysis
train_data.head()

survived_data = train_data.loc[train_data['Survived']==1,:]
died_data = train_data.loc[train_data['Survived']==0, :]

# plot the effect of sex vs. survival
# first splitting the data on sex, i.e. between male and female, using Boolean masking:
survived_male_data = survived_data.loc[survived_data['Sex']=='male',:]
died_male_data = died_data.loc[died_data['Sex']=='male',:]
survived_female_data = survived_data.loc[survived_data['Sex']=='female',:]
died_female_data = died_data.loc[died_data['Sex']=='female',:]

# Total number of females/males that survived and that died
# Split the survived and died data between male and female
survived_male_data = survived_data.loc[survived_data['Sex']=='male',:]
died_male_data = died_data.loc[died_data['Sex']=='male',:]
survived_female_data = survived_data.loc[survived_data['Sex']=='female',:]
died_female_data = died_data.loc[died_data['Sex']=='female',:]

# Total number of (fe)males that survived and that died
survived_male_n = survived_male_data.shape[0]
died_male_n = died_male_data.shape[0]
survived_female_n = survived_female_data.shape[0]
died_female_n = died_female_data.shape[0]

import matplotlib.pyplot as pp
# Bar plot drawing
fig, axes = plt.subplots(nrows=1, ncols=2)

# Sex vs. survival in total numbers
pp.axes(axes[0])
survived = pp.bar([0.5, 3.5], [survived_male_n, survived_female_n], width=1, color='#3BB200')
died = pp.bar([1.5, 4.5], [died_male_n, died_female_n], width=1, 
color='gray')
pp.xticks([1,4], ('Male', 'Female'))
pp.ylabel('Number of passengers')
pp.legend((survived, died), ('Survived', 'Died'), loc=0, fontsize = 'medium')

# Sex vs. survival in fractions
pp.axes(axes[1]) # select the second subplot
survived_pct = pp.bar([0.5, 3.5], [survived_male_n/(survived_male_n+died_male_n), 
                               survived_female_n/(survived_female_n+died_female_n)], 
                      width=1, color='#3BB200')
died_pct = pp.bar([1.5, 4.5], [died_male_n/(survived_male_n+died_male_n), 
                           died_female_n/(survived_female_n+died_female_n)], 
                  width=1, color='gray')
pp.xticks([1,4], ('Male', 'Female'))
pp.ylabel('Fraction')
pp.legend((survived, died), ('Survived', 'Died'), fontsize = 'medium')

fig.suptitle('Sex vs. survival', fontsize = 'x-large', y=1.03)
pp.tight_layout()
pp.show()

# there are far more male passengers than female passengers in the training data, and the survival odds of males are much lower than those of females.

#  take a look at the effects of age on survival. The 'Age' column of our DataFrame contains NaN values, which will have to be filtered out in order to create the histogram. 
# This means that the following plots will be based on only the 714 non-NaN values in the 'Age' column. Because we will have to remove passengers with NaN values more often, we will define a function checkNans that will help us do this:

def checkNans(arr, arr2=None):
    mask_nan = pd.isnull(arr) # using pandas isnull to also operate
                              # on string fields
    if mask_nan.sum()>0:
        any_nan = True
    else:
        any_nan = False
    n_nan = mask_nan.sum()
    
    masked_arr = arr[~mask_nan]
    if arr2 is not None:
        masked_arr2 = arr2[~mask_nan]
    else: 
        masked_arr2 = None

    return any_nan, masked_arr, masked_arr2, n_nan, mask_nan

'''import seaborn as sns

survived_age = checkNans(survived_data['Age'])[1]
died_age = checkNans(died_data['Age'])[1]

# Create density plots using seaborn's kdeplot
sns.kdeplot(survived_age, label='Survived', color='green')
sns.kdeplot(died_age, label='Died', color='red')

# Create histograms using seaborn's histplot
sns.histplot(survived_age, label='Survived', color='green')
sns.histplot(died_age, label='Died', color='red')

# Add a legend and set plot title and labels
sns.legend(fontsize='medium', title='Survival')
sns.set(xlabel='Age [years]', ylabel='Fraction')

# Show the plot
pp.show()'''

# Extract age data of the survived and died passengers,
# check for nans
survived_age = checkNans(survived_data['Age'])[1]
died_age = checkNans(died_data['Age'])[1]

# Histogram
# Determine bin edges of the combined data so that you can use these in 
# the final histogram, in order to make sure that the histogram bin 
# widths are equal for both groups
stacked = np.hstack((survived_age, died_age))
bins = np.histogram(stacked, bins=16, range=(0,stacked.max()))[1] 

# Creating the histograms
survived = pp.hist(survived_age, bins, facecolor='green', alpha=0.5)
died = pp.hist(died_age, bins, facecolor='red', alpha=0.5)

# Create custom handles for adding a legend to the histogram
import matplotlib.patches as mpatches
survived_handle = mpatches.Patch(facecolor='green', alpha=0.5, 
                                label='Survived', edgecolor='black')
died_handle = mpatches.Patch(facecolor='red', alpha=0.5, label='Died', 
                                edgecolor='black')
pp.legend((survived_handle, died_handle), ('Survived', 'Died'), loc=0, 
                                fontsize = 'medium')

# Other plot settings
pp.title('Age vs. survival', fontsize = 'x-large', y=1.02)
pp.xlabel('Age [years]')
pp.ylabel('Fraction')
pp.tight_layout()
pp.show()


import matplotlib.pyplot as plt

# Extract age data of the survived and died passengers,
# check for nans
survived_age = checkNans(survived_data['Age'])[1]
died_age = checkNans(died_data['Age'])[1]

# Histogram
# Determine bin edges of the combined data so that you can use these in 
# the final histogram, in order to make sure that the histogram bin 
# widths are equal for both groups
stacked = np.hstack((survived_age, died_age))
bins = np.histogram(stacked, bins=16, range=(0,stacked.max()))[1] 

# Creating the histograms
n_survived = survived_age.size
n_died = died_age.size
survived = plt.hist(survived_age, bins, weights=np.ones(n_survived)/(n_survived+n_died), facecolor='green', alpha=0.5)
died = plt.hist(died_age, bins, weights=np.ones(n_died)/(n_survived+n_died), facecolor='red', alpha=0.5)

# Create custom handles for adding a legend to the histogram
import matplotlib.patches as mpatches
survived_handle = mpatches.Patch(facecolor='green', alpha=0.5, 
                                label='Survived', edgecolor='black')
died_handle = mpatches.Patch(facecolor='red', alpha=0.5, label='Died', 
                                edgecolor='black')
pp.legend((survived_handle, died_handle), ('Survived', 'Died'), loc=0, 
                                fontsize = 'medium')

# Other plot settings
pp.title('Age vs. survival', fontsize = 'x-large', y=1.02)
pp.xlabel('Age [years]')
pp.ylabel('Fraction')
pp.tight_layout()
pp.show()



'''
So what we can see here is that the age distributions of those who survived and those who died are quite similar. Passengers older than ~65 seem to have worse odds of surviving, and children aged below ~16 seem to have better odds of surviving, but overall the effects are not too great. It would be interesting to replicate this for males and females separately, which is what we'll do next.
'''
'''# Extract age data of the survived and died passengers for males 
# and females separately, check for nans
survived_male_age = checkNans(survived_male_data['Age'])[1]
died_male_age = checkNans(died_male_data['Age'])[1]
survived_female_age = checkNans(survived_female_data['Age'])[1]
died_female_age = checkNans(died_female_data['Age'])[1]

# Histogram
# Create subplots with shared y-axis
fig, axes = pp.subplots(nrows=1, ncols=2, figsize=(8,4), sharey=True)

# Creating the histograms
# For the bin edges, we can use the same as bin edges as in the 
# previous plot
# Male histogram
pp.axes(axes[0])
survived_male = pp.hist(survived_male_age, bins,  
                        facecolor='green', alpha=0.5)
died_male = pp.hist(died_male_age, bins,  facecolor='red', 
                        alpha=0.5)
pp.legend((survived_handle, died_handle), ('Survived', 'Died'),
          loc=0, fontsize = 'medium') # Using the same legend handles 
                                      # as before
pp.title('Male')
pp.xlabel('Age [years]')
pp.ylabel('Fraction')
pp.xlim([0,stacked.max()]) # Using the same range as in the previous plot
pp.tight_layout()

# Female histogram
pp.axes(axes[1])
survived_female = pp.hist(survived_female_age, bins, 
                          facecolor='green', alpha=0.5)
died_female = pp.hist(died_female_age, bins, facecolor='red', 
                      alpha=0.5)
pp.legend((survived_handle, died_handle), ('Survived', 'Died'), loc=0, 
          fontsize = 'medium') # Using the same legend handles as before
pp.title('Female')
pp.xlabel('Age [years]')
pp.xlim([0,stacked.max()]) # Using the same range as in the previous plot
pp.tight_layout()
fig.suptitle('Age vs. survival', fontsize = 'x-large', y=1.02)
pp.show()'''


import matplotlib.pyplot as plt

# Extract age data of the survived and died passengers for males 
# and females separately, check for nans
survived_male_age = checkNans(survived_male_data['Age'])[1]
died_male_age = checkNans(died_male_data['Age'])[1]
survived_female_age = checkNans(survived_female_data['Age'])[1]
died_female_age = checkNans(died_female_data['Age'])[1]

# Histogram
# Create subplots with shared y-axis
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4), sharey=True)

# Creating the histograms
# For the bin edges, we can use the same as bin edges as in the 
# previous plot
# Male histogram
plt.axes(axes[0])
n_survived_male = survived_male_age.size
n_died_male = died_male_age.size
survived_male = plt.hist(survived_male_age, bins, weights=np.ones(n_survived_male)/n_survived_male,
                        facecolor='green', alpha=0.5)
died_male = plt.hist(died_male_age, bins, weights=np.ones(n_died_male)/n_died_male,
                        facecolor='red', alpha=0.5)
plt.legend((survived_handle, died_handle), ('Survived', 'Died'),
          loc=0, fontsize = 'medium') # Using the same legend handles 
                                      # as before
plt.title('Male')
plt.xlabel('Age [years]')
plt.ylabel('Fraction')
plt.xlim([0,stacked.max()]) # Using the same range as in the previous plot
plt.tight_layout()

# Female histogram
plt.axes(axes[1])
n_survived_female = survived_female_age.size
n_died_female = died_female_age.size
survived_female = plt.hist(survived_female_age, bins, weights=np.ones(n_survived_female)/n_survived_female,
                          facecolor='green', alpha=0.5)
died_female = plt.hist(died_female_age, bins, weights=np.ones(n_died_female)/n_died_female,
                      facecolor='red', alpha=0.5)
plt.legend((survived_handle, died_handle), ('Survived', 'Died'), loc=0, 
          fontsize = 'medium') # Using the same legend handles as before
plt.title('Female')
plt.xlabel('Age [years]')
plt.xlim([0,stacked.max()]) # Using the same range as in the previous plot
plt.tight_layout()
fig.suptitle
pp.show()


# The added differentation between male and female does not provide any real insights except that for really young (<10 years) male childs the odds of surviving are much higher than for girls of the same age. This could be a statistical glitch if the number of boys aged <10 is much different from the number of girls aged <10, or if both groups are very small in size. However, both groups have 19 members.

print('Number of males aged <10: {}'.format\
      (survived_male_age[survived_male_age<10].count()))
print('Number of females aged <10: {}'.format\
      (survived_female_age[survived_female_age<10].count()))

# Ticket class and ticket fare most likely will somewhat move together. Let's make some boxplots to verify this.

train_data.head()
# Extract the ticket fares for the various ticket classes
fare_1 = train_data['Fare'].loc[train_data['Pclass']==1]
fare_2 = train_data['Fare'].loc[train_data['Pclass']==2]
fare_3 = train_data['Fare'].loc[train_data['Pclass']==3]
# Boxplots
pp.boxplot([fare_1, fare_2, fare_3])
pp.ylim((0,train_data['Fare'].quantile(0.98))) 
# Only the lowest 98% of fares are shown because 
# otherwise the boxes would be hard to compare visually
# Plot settings
pp.ylabel('Fare')
pp.xlabel('Ticket class')
pp.title('Ticket class vs. fare', fontsize = 'x-large', y=1.02)
pp.tight_layout()
pp.show()


# Histogram
# Determine bin edges of the combined data so that you can use 
# these in the final histogram, in order to make sure that the 
# histogram bin widths are equal for all three classes
bins = np.histogram(train_data['Fare'].loc[train_data['Fare']
        <(train_data['Fare'].quantile(0.98))], bins=20, 
        range=(0,train_data['Fare'].loc[train_data['Fare']<
        (train_data['Fare'].quantile(0.98))].max()))[1] 

# Creating the histograms
class_1 = pp.hist(fare_1, bins, facecolor='blue', alpha=0.5)
class_2 = pp.hist(fare_2, bins, facecolor='red', alpha=0.5)
class_3 = pp.hist(fare_3, bins, facecolor='green', alpha=0.5)

# Create custom handles for adding a legend to the histogram
class_1_handle = mpatches.Patch(facecolor='blue', alpha=0.5, 
                                label='Class 1', edgecolor='black')
class_2_handle = mpatches.Patch(facecolor='red', alpha=0.5, 
                                label='Class 2', edgecolor='black')
class_3_handle = mpatches.Patch(facecolor='green', alpha=0.5, 
                                label='Class 3', edgecolor='black')

pp.legend((class_1_handle, class_2_handle, class_3_handle), 
          ('1st Class', '2nd Class', '3rd Class'), 
          loc=0, fontsize = 'medium')

# Plot settings
pp.title('Ticket class vs. fare', fontsize = 'x-large', y=1.02)
pp.xlabel('Fare')
pp.ylabel('Fraction')
pp.xlim((0,train_data['Fare'].quantile(0.98)))
pp.tight_layout()
pp.show()

#　So the classes somewhat overlap, although the histogram does not visualize this very clearly. Let's look at the effect of ticket class on survival.

# Extract the survival data for the various ticket classes
survived_class_1 = survived_data['Survived'].loc[survived_data['Pclass']==1]
died_class_1 = died_data['Survived'].loc[died_data['Pclass']==1]
survived_class_2 = survived_data['Survived'].loc[survived_data['Pclass']==2]
died_class_2 = died_data['Survived'].loc[died_data['Pclass']==2]
survived_class_3 = survived_data['Survived'].loc[survived_data['Pclass']==3]
died_class_3 = died_data['Survived'].loc[died_data['Pclass']==3]

# Total number of passengers from each class that survived and that died
survived_class_1_n = survived_class_1.shape[0]
died_class_1_n = died_class_1.shape[0]
survived_class_2_n = survived_class_2.shape[0]
died_class_2_n = died_class_2.shape[0]
survived_class_3_n = survived_class_3.shape[0]
died_class_3_n = died_class_3.shape[0]

# Bar plot drawing
fig, axes = pp.subplots(nrows=1, ncols=2)

# Ticket class vs. survival in total numbers
pp.axes(axes[0])
survived = pp.bar([0.5, 3.5, 6.5], [survived_class_1_n, survived_class_2_n,
                                    survived_class_3_n], width=1, 
                                    color='#3BB200')
died = pp.bar([1.5, 4.5, 7.5], [died_class_1_n, died_class_2_n, 
                                died_class_3_n], width=1, color='red')
pp.xticks([1.5,4.5,7.5], ('1st Class', '2nd Class', '3rd Class'))

pp.ylabel('Number of passengers')
pp.legend((survived, died), ('Survived', 'Died'), loc=0, 
          fontsize = 'medium')

# Sex vs. survival in fractions
pp.axes(axes[1])
survived_pct = pp.bar([0.5, 3.5, 6.5], [survived_class_1_n/(survived_class_1_n + died_class_1_n), 
                                        survived_class_2_n/(survived_class_2_n + died_class_2_n),
                                        survived_class_3_n/(survived_class_3_n + died_class_3_n)], 
                                        width=1, color='#3BB200')
died_pct = pp.bar([1.5, 4.5, 7.5], [died_class_1_n/(survived_class_1_n + died_class_1_n), 
                                        died_class_2_n/(survived_class_2_n + died_class_2_n),
                                        died_class_3_n/(survived_class_3_n + died_class_3_n)],  
                                        width=1, color='red')
pp.xticks([1.5,4.5,7.5], ('1st Class', '2nd Class', '3rd Class'))
#pp.xlim([0,8])
pp.ylabel('Fraction')
leg = pp.legend((survived, died), ('Survived', 'Died'), 
                fontsize = 'medium', loc='upper left')
fig.suptitle('Ticket class vs. survival', fontsize = 'x-large', y=1.03)
pp.tight_layout()
pp.show()             

# So it seems that 1st class passengers had better odds of survival, and 3rd class passengers had far worse odds. 2nd Class passengers were about equally likely to either survive or die.

# The next thing that we'll look at is the number of siblings+spouse vs. survival and the number of parents+children vs. survival. The feature is discrete and numerical; we will therefore bin the data in bins of width 1 and create a histogram like above. The 'SibSp' and 'Parch' columns of our data do not contain any NaN values so we do not need to preprocess them.

fig, axes = pp.subplots(nrows=1, ncols=2, figsize=(8,4), sharey=True)
# SIBSP
pp.axes(axes[0])
# Extract SibSp data of the survived and died passengers
survived_sibsp = survived_data['SibSp']
died_sibsp = died_data['SibSp']

# Histogram
# Determine bin edges of the combined data so that you can use 
# these in the final histogram, in order to make sure that the 
# histogram bin widths are equal for both groups
stacked = np.hstack((survived_sibsp, died_sibsp))
bins = np.histogram(stacked, bins=stacked.max()+1, range=
                    (0,stacked.max()+1))[1] 
# The number of bins = stacked.max() so that each bin is of width 1

# Creating the histograms
survived = pp.hist(survived_sibsp, bins,  facecolor='green', 
                   alpha=0.5)
died = pp.hist(died_sibsp, bins,  facecolor='red', alpha=0.5)
# Plot settings
pp.legend((survived_handle, died_handle), ('Survived', 'Died'), loc=0, 
          fontsize = 'medium') # Using the same legend handles as before
pp.title('No. siblings+spouses vs. survival', fontsize = 'x-large', y=1.02)
pp.xlabel('No. siblings+spouses')
pp.ylabel('Fraction')
pp.xticks(np.arange(train_data['SibSp'].max()+1)+0.5, 
          np.arange(train_data['SibSp'].max()+1))
pp.xlim([0,stacked.max()+1])
pp.tight_layout()

# PARCH
pp.axes(axes[1])
# Extract Parch data of the survived and died passengers
survived_parch = survived_data['Parch']
died_parch = died_data['Parch']

# Histogram
# Determine bin edges of the combined data so that you can use 
# these in the final histogram, in order to make sure that the 
# histogram bin widths are equal for both groups
stacked = np.hstack((survived_parch, died_parch))
bins = np.histogram(stacked, bins=stacked.max(), 
                    range=(0,stacked.max()))[1] 
    # The number of bins = stacked.max() so that each bin is of width 1

# Creating the histograms
survived = pp.hist(survived_sibsp, bins,  facecolor='green', 
                   alpha=0.5)
died = pp.hist(died_sibsp, bins, facecolor='red', alpha=0.5)

# Plot settings
pp.legend((survived_handle, died_handle), ('Survived', 'Died'), loc=0, 
          fontsize = 'medium') # Using the same legend handles as before
pp.title('No. parents+children vs. survival', fontsize = 'x-large', y=1.02)
pp.xlabel('No. parents+children')
pp.xticks(np.arange(train_data['Parch'].max()+1)+0.5, 
          np.arange(train_data['Parch'].max()+1))
pp.ylabel('Fraction')
pp.xlim([0,stacked.max()+1])
pp.tight_layout()

print('Fraction of total survived / died passengers \n\
accounted for by a certain bin')
pp.show()

# SIBSP
## The same as above, but now with the fractions within each category:
# Extract the survival data for the various SibSp classes
sibsp_classes = np.arange(train_data['SibSp'].max() + 1)
sibsp_data = {'survived_n':np.array([]), 'survived_pct':np.array([]), 
              'died_n':np.array([]), 'died_pct':np.array([])}
for ii in sibsp_classes:
    sibsp_data['survived_'+str(ii)] = survived_data['Survived'].loc[
        survived_data['SibSp']==ii]
    sibsp_data['survived_n'] = np.append(sibsp_data['survived_n'],
        np.array((sibsp_data['survived_'+str(ii)].count())))
    sibsp_data['died_'+str(ii)] = died_data['Survived'].loc[
        died_data['SibSp']==ii]
    sibsp_data['died_n'] = np.append(sibsp_data['died_n'],
        np.array((sibsp_data['died_'+str(ii)].count())))

sibsp_data['survived_pct'] = (sibsp_data['survived_n']/  
                                              (sibsp_data['survived_n']+
                                               sibsp_data['died_n']))
sibsp_data['survived_pct'][np.isnan(sibsp_data['survived_pct'])]=0
sibsp_data['died_pct'] = (sibsp_data['died_n']/  
                                              (sibsp_data['survived_n']+
                                               sibsp_data['died_n']))
sibsp_data['died_pct'][np.isnan(sibsp_data['died_pct'])]=0

# No. of siblings+spouses vs. survival in fractions

survived_pct = pp.bar(np.arange(train_data['SibSp'].max()+1)*3+0.5, 
                      sibsp_data['survived_pct'], width=1, color='#3BB200')
died_pct = pp.bar(np.arange(train_data['SibSp'].max()+1)*3+1.5, 
                      sibsp_data['died_pct'], width=1, color='red')
pp.xticks(np.arange(train_data['SibSp'].max()+1)*3+1.5, 
          np.arange(train_data['SibSp'].max()+1))
pp.xlim([0,(train_data['SibSp'].max()+1)*3])
pp.xlabel('No. of siblings+spouse')
pp.ylabel('Fraction')
leg = pp.legend((survived_pct, died_pct), ('Survived', 'Died'), 
                fontsize = 'medium', loc='upper left')
pp.title('No. of siblings+spouse vs. survival', fontsize = 
         'x-large', y=1.03)
pp.tight_layout()
print('Fraction of survived / died passengers within a certain bin')
pp.show()

# PARCH
# Extract ParCh data of the survived and died passengers
survived_parch = survived_data['Parch']
died_parch = died_data['Parch']

# Extract the survival data for the various ParCh classes
parch_classes = np.arange(train_data['Parch'].max() + 1)
parch_data = {'survived_n':np.array([]), 'survived_pct':np.array([]), 
              'died_n':np.array([]), 'died_pct':np.array([])}
for ii in parch_classes:
    parch_data['survived_'+str(ii)] = survived_data['Survived'].loc[
        survived_data['Parch']==ii]
    parch_data['survived_n'] = np.append(parch_data['survived_n'],
        np.array((parch_data['survived_'+str(ii)].count())))
    parch_data['died_'+str(ii)] = died_data['Survived'].loc[
        died_data['Parch']==ii]
    parch_data['died_n'] = np.append(parch_data['died_n'],
        np.array((parch_data['died_'+str(ii)].count())))

parch_data['survived_pct'] = (parch_data['survived_n']/  
                                              (parch_data['survived_n']+
                                               parch_data['died_n']))
parch_data['survived_pct'][np.isnan(parch_data['survived_pct'])]=0
parch_data['died_pct'] = (parch_data['died_n']/  
                                              (parch_data['survived_n']+
                                               parch_data['died_n']))
parch_data['died_pct'][np.isnan(parch_data['died_pct'])]=0

# No. of parents+children vs. survival in fractions
survived_pct = pp.bar(np.arange(train_data['Parch'].max()+1)*3+0.5, 
                      parch_data['survived_pct'], width=1, color='#3BB200')
died_pct = pp.bar(np.arange(train_data['Parch'].max()+1)*3+1.5, 
                      parch_data['died_pct'], width=1, color='red')
pp.xticks(np.arange(train_data['Parch'].max()+1)*3+1.5, 
          np.arange(train_data['Parch'].max()+1))
pp.xlim([0,(train_data['Parch'].max()+1)*3])
pp.xlabel('No. of parents+children')
pp.ylabel('Fraction')
leg = pp.legend((survived_pct, died_pct), ('Survived', 'Died'), 
                fontsize = 'medium', loc='upper left')
pp.title('No. of parents+children vs. survival', fontsize = 'x-large', 
         y = 1.03)
pp.tight_layout()
print('Fraction of survived / died passengers within a certain bin')
pp.show()
'''
We have already seen that the ticket fares for the three ticket classes do overlap, and that the ticket class has predictive power on survival odds. When we group the ticket fares into bins, we can show the survival odds for each bin. Below I have grouped the data in bins of width=10; above a ticket fare of 250 only a few data points exist, and these have been grouped together.'''

bins = np.append(np.arange(0,251,10), train_data['Fare'].max())
train_fare_binned = pd.cut(train_data['Fare'], bins, include_lowest=True)
train_data[['Survived']].groupby(train_fare_binned).mean()

print('Number of passengers:\n{}'.format(train_data[
    'Fare'].groupby(train_fare_binned).count()))
print('\n')
print('Average survival:\n{}'.format(train_data[
    'Survived'].groupby(train_fare_binned).mean()))

'''
We observe that passengers in the lowest fare category are very numerous and also have very bad survival odds. We further notice that the next two most-populous groups have better survival odds than for our training set as a whole (38.4%).

Lastly, the average survival rates for the various categories in the 'Embarked', 'Cabin_cat' and 'Title' features will be presented.'''

print('Average survival:\n{}'.format(train_data[['Embarked',
    'Survived']].groupby('Embarked').mean()))
print('\n')
print('{}'.format(train_data[['Cabin_cat',
    'Survived']].groupby('Cabin_cat').mean()))
print('\n')
print('{}'.format(train_data[['Title',
    'Survived']].groupby('Title').mean()))

'''There seems to be enough distribution within those features for them to be relevant to our classification problem, so they will be included.

We have seen some trends in the data. If we assume independence of the examined features, a man aged 20-25, travelling 3rd class, with a total of 5 siblings + spouse, and 4 parents + children, boarded in Southampton, would have far worse odds of surviving than a married woman aged 35-40, travelling 1st class, with 1 sibling or spouse, and 3 parents + children, boarded in Cherbourg.

This concludes the Exploratory Data Analysis that we will be doing. We will now focus on training machine learning models on the data.'''

# Preprocessing for machine learning

train_data[['Sex','Embarked','Title','CabinCat']].info()
train_data.head()

# So only the 'Embarked' feature contains null entries, 2 in total. Let's find those entries:
train_data.loc[train_data['Embarked'].isnull()]

from sklearn.preprocessing import LabelEncoder
# Converting to numerical features
# Sex feature
le_sex = LabelEncoder()
sex_numerical = le_sex.fit_transform(train_data['Sex'])
sex_numerical_classes = le_sex.classes_

# Title feature
le_title = LabelEncoder()
title_numerical = le_title.fit_transform(train_data['Title'])
title_numerical_classes = le_title.classes_

# Cabin_cat feature
le_cabin_cat = LabelEncoder()
cabin_cat_numerical = le_cabin_cat.fit_transform(train_data['CabinCat'])
cabin_cat_numerical_classes = le_cabin_cat.classes_

print('Classes of Sex feature:\n{}\n{}'.format(
        np.arange(len(sex_numerical_classes)), sex_numerical_classes))
print('')
print('Classes of Title feature:\n{}\n{}'.format(
        np.arange(len(title_numerical_classes)), title_numerical_classes))
print('')
print('Classes of Cabin_cat feature:\n{}\n{}'.format(
        np.arange(len(cabin_cat_numerical_classes)), cabin_cat_numerical_classes))

# We will create One-Hot labeled features:
from sklearn.preprocessing import OneHotEncoder
# Sex feature
enc_sex = OneHotEncoder(sparse=False)
sex_onehot = enc_sex.fit_transform(sex_numerical.reshape(-1,1))
# Title feature
enc_title = OneHotEncoder(sparse=False)
title_onehot = enc_title.fit_transform(title_numerical.reshape(-1,1))

# Cabin_cat feature
enc_cabin_cat = OneHotEncoder(sparse=False)
cabin_cat_onehot = enc_cabin_cat.fit_transform(cabin_cat_numerical.reshape(-1,1))
'''
Drop the original categorical features and add the one-hot labeled features. Map names to the new features based on the classes printed above, using a new function pdAssignWithOHLabel():'''

def pdAssignWithOHLabel(df, column, onehot_labeled, class_labels):
    to_assign = {}
    for c_idx, label in enumerate(class_labels):
        to_assign[column+'_'+label] = onehot_labeled[:,c_idx]
    df = df.assign(**to_assign)
    return df

# Sex feature
train_data = pdAssignWithOHLabel(train_data, 'Sex', 
                                 sex_onehot, sex_numerical_classes)
train_data = train_data.drop('Sex',axis=1)

# Title feature
train_data = pdAssignWithOHLabel(train_data, 'Title', 
                                 title_onehot, title_numerical_classes)
train_data = train_data.drop('Title',axis=1)

# Cabin_cat feature
train_data = pdAssignWithOHLabel(train_data, 'Cabin_cat', 
                            cabin_cat_onehot, cabin_cat_numerical_classes)
train_data = train_data.drop('Cabin_cat',axis=1)

train_data.head()

'''
The 'Embarked' data has not been imputed or Hot-One labeled yet. We will perform imputation based on the 5 nearest neighbors of a passenger; accurate nearest neighbor finding requires our features to be scaled. sklearn.preprocessing.StandardScaler() will be used to scale the features; however, StandardScaler does not provide good results when the data contains outliers. We recall the ticket fare box plot from the Exploratory Data Analysis section; only the smallest 98% of the data was shown in that plot, because there were some far-outlying points. We plot the ticket fare data again:'''

pp.boxplot([train_data['Fare']]) 
pp.show()


'''
Data points that lie outside a range of mean + 3* standard deviation or even farther away can certainly bring meaningul information to our models; however, in case of the 'Fare' feature, there are 3 data points that unproportionally influence the results of the StandardScaler because they lie so far apart from the other values. Let's set their values to the mean+5* standard deviation of the ticket fares:'''

mu = train_data['Fare'].mean()
sd = train_data['Fare'].std()

row_mask = train_data['Fare']>mu+5*sd
# Update the Fare value for these rows to the mean plus 5 times the standard deviation
train_data.loc[row_mask, 'Fare'] = mu + 5*sd

'''
Now we can perform standard scaling on all features except the 'Embarked' and 'Age' feature, because both need to be imputed. This scaling will be performed on a temporary copy of the training data because with the sole purpose of being able to more accurately find nearest neighbors for data imputation. Persistent scaling will be performed on the training data in the ML fitting section. Also, the 'Survived' feature does not need scaling since it will be our target label in the ML model training:'''

from sklearn.preprocessing import StandardScaler
sc_tmp = StandardScaler()
tmp_scaled = train_data.copy().drop(['Embarked','Age','Survived'], axis=1) # create a copy of the data
tmp_scaled = pd.DataFrame(sc_tmp.fit_transform(tmp_scaled),columns=tmp_scaled.columns, index=tmp_scaled.index)

# Add the non-scaled features to this temporary DataFrame
tmp_scaled = tmp_scaled.assign(Survived=train_data['Survived'])
tmp_scaled = tmp_scaled.assign(Embarked=train_data['Embarked'])
tmp_scaled = tmp_scaled.assign(Age=train_data['Age'])

tmp_scaled.head()
'''
We found before that passengers 62 and 830 did not have non-null values for the 'Embarked' feature. Here, let's find the 5 nearest neighbors of these passengers based on all features except 'Age', 'Embarked' and 'Survived', and assign a value for 'Embarked' based on the average value of their nearest neighbors on that feature:'''


from sklearn.neighbors import KDTree
tmp = tmp_scaled.copy().drop(['Survived','Age','Embarked','CabinCat'], axis=1).values

print(tmp)

# find all string columns
string_cols = [col for col in tmp_scaled.columns if tmp_scaled[col].dtype == 'object']
string_cols


row_idx = pd.isnull(train_data['Embarked'])
tree = KDTree(tmp)
dist, ind = tree.query(tmp[[62, 830]], k=6) 
# The k nearest neighbors include the passenger itself, 
# so we specify k=6 to get the 5 nearest neighbors
for i in ind:
    print('5 closest neigbors to passenger {} and their values for Embarked:\n{}\n'\
          .format(i[0], train_data['Embarked'].loc[i[1:]]))

# Based on the above, both passengers will be assigned an 'S' in the 'Embarked' feature:

train_data.loc[[62, 830], 'Embarked'] = 'S'









