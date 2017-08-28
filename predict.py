from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib
matplotlib.use('Gtk3Agg') #work around for matplotlib in Python3
import matplotlib.pyplot as plt
import seaborn as sb

#######
"""
This script is for the Blood Donation challenge from DrivenData. My username on that site is 'muddy_gate' and my best score is 0.4451 (top 10%).


The goal is to predict whether or not a person will donate blood in March of 2007. Several different approaches are taken and compared. DrivenData does not provide the actual donation figures for the test group; predictions must be submitted and then scored with a log loss metric.  More details can be found at:

https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/

This script takes the following steps:

Characterize the data
Identify/create key factors
Predict donors:
  Random Forest 
  Logistic Regression (current best score of the three)
  K-Nearest Neighbors
"""
#######


#######
#load testing and training csv into panda.DataFrame
#######
train = pd.read_csv("./data/training.csv")
test = pd.read_csv("./data/test.csv")
train = pd.DataFrame(train)
test = pd.DataFrame(test)
print(test.head())

#######
"""
Create new variables 'Span' and 'Rate': 
Span = Months since First Donation - Months since Last Donation
Rate = Span/Number of Donations
Will only use Rate in final Fit
"""
#######

train.insert(len(train.columns)-1, "Span", (train['Months since First Donation'] - train['Months since Last Donation']))
train.insert(len(train.columns)-2, "Rate", train['Span']/train['Number of Donations'])
test.insert(len(test.columns)-1, "Span", (test['Months since First Donation'] - train['Months since Last Donation']))
test.insert(len(test.columns)-2, "Rate", test['Span']/test['Number of Donations'])

#Create a scatter plot of all feature variables provided, as well as target variable

axs = scatter_matrix(train,alpha=0.2, figsize=(7,7), diagonal='hist')
n = len(train.columns)
for x in range(n):
    for y in range(n):
        #to get the axis of subplots and rotate the label names, for legibility
        ax = axs[x, y]
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        # to make sure y axis names are outside the plot area
        if y==0: ax.yaxis.labelpad = 75

    
plt.tight_layout(h_pad=0,w_pad=0)
plt.savefig('./scatter.png')
plt.close()

######
"""
'Number of Donations' and 'Total Volume Donated' are exactly correlated; i.e. Volume per Donation is a constant. For training and prediction, only 'Number of Donations' is used. The calculated 'Span' variable is strongly correlated with 'Months Since First Donation' and 'Rate', and it is not used either.
"""
######


#######
#Select features to be used for training and prediction
#######

cols = [col for col in train.columns if col not in [ 'Unnamed: 0','Total Volume Donated (c.c.)','Span','Made Donation in March 2007']]
#print(cols)
features = train[cols]
features = features.columns[:]
#print(features)

#Seaborn heatmap for training dataset: use dataframe with just relevant features

corr_mat = train[features].corr()
sb.heatmap(corr_mat)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tick_params(labelsize=4)
plt.savefig('./heatmap.png')
plt.close()


#Declare training output values

march = train['Made Donation in March 2007']
#print(march)



#######
#Create RF Classifier: run with selected features from train on the "Made Donation in March' column
#######

clf = RandomForestClassifier(n_jobs=2)
clf.fit(train[features],march)

clf.predict(test[features])

#predprob = clf.predict_proba(test[features])
#print(predprob)
#print(predprob[:,[1]])


######
#Can't create confusion matrix since we do not have the actual donation numbers
#####

#confusion = pd.crosstab(test['Made Donation in March 2007'],predict, rownames = "Actual", colmanes = 'Predicted')
#print(confusion)


#######
#Create a logistical regression predictor using the same features as random forest
#######

log = LogisticRegression()
log.fit(train[features],march)

print(log.score(train[features],march))
print(march.mean())

predprob = log.predict_proba(test[features])




#######
#Save output as 'Donor' - 'Probability': create from numpy array predprob, and panda Series test['Unnamed: 0'] to match required input form
#######

name = ['Made Donation in March 2007']
final = pd.DataFrame(predprob[:,[1]],columns=name)
final[""] = test['Unnamed: 0']
colnames = final.columns.tolist()
colnames.insert(1, colnames.pop(colnames.index("Made Donation in March 2007")))
final = final.reindex(columns=colnames)
#print(final)

print(list(zip(train[features], clf.feature_importances_)))

final.to_csv('blood.csv',sep=',', index=False)
