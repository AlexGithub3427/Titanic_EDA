import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Thesis: 
# Women will have a higher survival rate than men. Youth will have a higher survival rate than adults.
# Rich individuals will have a higher survival rate than poor. 


df = pd.read_csv('Titanic-Dataset.csv.xls')

#CLEANING EMPTY DATA VALUES
# Roughly 20% of 'Age' column missing, replace empty cells with median (skewed)
age_median = df['Age'].median()
df.fillna({'Age': age_median}, inplace = True)
# roughly 77% of 'Cabin'(number) column missing, remove column entirely
df.drop('Cabin', axis = 1, inplace = True)
# only two entries of 'Embarked'(port) column missing (~0.2%), replace with mode (categorical)
embarked_mode = df['Embarked'].mode()[0]
df.fillna({'Embarked': embarked_mode}, inplace = True)

#REMOVING NON-ESSENTIAL INFORMATION
# 'PassengerId' column has no correlation with survival, remove column entirely
df.drop('PassengerId', axis = 1, inplace = True)
# 'Ticket'(number) column messy and inconsequential to analysis, remove column entirely
df.drop('Ticket', axis = 1, inplace = True )

#DATA EXTRACTION
# We can extrapolate titles from 'Name' column that likely correlates with survival
# Doing so, we create a new column 'Title' and remove the 'Name' column irrelevant to survival
df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\s*\.')
# Rare titles denoting nobility, military, etc. will be designated as 'Rare'
rare_titles = ['Dr', 'Rev', 'Major', 'Col', 'the Countess', 'Capt', 'Sir', 'Lady', 'Don', 'Jonkheer']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
# Convert French abbreviations (and Ms) to English
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
# We can now drop the 'Name' column as it is no longer necessary for analysis
df.drop('Name', axis = 1, inplace = True)

#DATA PRESENTATION AND ANALYSIS
sns.set_theme()
fig, axes = plt.subplots(2, 4, figsize = (8, 6))
plt.tight_layout()

sns.barplot(data = df, x = 'Sex', y = 'Survived', ax = axes[0, 0])
axes[0, 0].set_title('Survival Rate by Sex')

sns.histplot(data = df, x = 'Age', hue = 'Survived', kde = True, ax = axes[1, 0])
axes[1, 0].set_title('Survival Rate by Age')

sns.barplot(data = df, x = 'Pclass', y = 'Survived', ax = axes[0, 1])
axes[0, 1].set_title('Survival Rate by Ticket Class')

sns.histplot(data = df, x = 'Fare', hue = 'Survived', kde = True, ax = axes[1, 1])
axes[1, 1].set_title('Survival Rate by Fare Cost')

sns.barplot(data = df, x = 'SibSp', y = 'Survived', ax = axes[0, 2])
axes[0, 2].set_title('Survival Rate by No. Siblings/Spouses Aboard')

sns.barplot(data = df, x = 'Parch', y = 'Survived', ax = axes[1, 2])
axes[1, 2].set_title('Survival Rate by No. Parents/Children Aboard')

sns.barplot(data = df, x = 'Title', y = 'Survived', ax = axes[0, 3])
axes[0, 3].set_title('Survival Rate by Title')

sns.barplot(data = df, x = 'Embarked', y = 'Survived', ax = axes[1, 3])
axes[1, 3].set_title('Survival Rate by Embarked Location')

plt.show()

# Conclusion:
# Women had significantly greater survival rates. Based on ticket class and fare cost, 
# wealth correlated with greater survival as well. Adult ages (20 - 40) showed significantly
# worse survival rates than younger and older ages. Number of parents and children showed an
# inconclusive relation to survival rate. Number of siblings/spouses did show a trend (greatest
# survival at 1 with linear decrease to 4). The Titles graph further emphasized trends we have 
# established as we see Mrs, Miss, and Master (referring to adult women, girls, and boys) having
# exceptional survival rates, while Mr (adult men) having drastically lower survival rates. Rare
# titles surprising did not signficantly correlate with a great survival rate as it neared 35% survival.
# Finally, those embarking from the port of Cherbourg(C) had a significantly greater survival rate
# than those embarking from the port of Queenstown(Q) and Southhampton(S). Thus, alongside other
# findings, our thesis is correct.