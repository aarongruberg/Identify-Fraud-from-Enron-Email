#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt

from helper import find_max, computePercent

from sklearn.feature_selection import SelectKBest


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


### COMPLETE FEATURES LIST excluding 'email_address'
#features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', \
#'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', \
#'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', \
#'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', \
#'from_this_person_to_poi', 'shared_receipt_with_poi']


### TOP 4 FEATURES from SelectKBest scores on the complete features list
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### DATA PROFILE
c = 0
total_c = 0
poi = 0
not_poi = 0
for person in data_dict.keys():
	c += 1
	feature_c = 0
	
	if data_dict[person]['poi'] == True:
		poi += 1

	else: 
		not_poi += 1

	for feature in data_dict[person]:
		feature_c += 1
		total_c += 1

print "\n"
print "Data Profile"
print "-----------------------------------------------------------------"
print "Number of employees:", c
print "Number of features per employee:", feature_c 
print "Total number of data points before feature selection:", total_c 
print "Total pois:", poi 
print "Total not pois:", not_poi
print "\n"


### Task 2: Remove outliers

# many outliers in this dataset are important and should be kept
# find_max revealed 'TOTAL' to have the max value
# for financial features.  Remove the 'TOTAL' key.
# There was another non person key 'THE TRAVEL AGENCY IN THE PARK' 
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# find the max/min value and person for a feature
#print find_max(data_dict, 'bonus')


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

for person in data_dict.keys():
	# new feature 'percent_exercised_stock'
	num = data_dict[person]['exercised_stock_options']
	den = data_dict[person]['total_stock_value']
	data_dict[person]['percent_exercised_stock'] = computePercent(num, den)
	
	# new feature 'percent_bonus'
	num = data_dict[person]['bonus']
	den = data_dict[person]['salary']
	data_dict[person]['percent_bonus'] = computePercent(num, den)



### Counting NaN values for selected features
print "NaN values:"
print "-----------------------------------------------------------------"
poi_nan = 0
exercised_stock_options_nan = 0
total_stock_value_nan = 0
bonus_nan = 0
salary_nan = 0

for person in data_dict.keys():
	if data_dict[person]['poi'] == 'NaN':
		poi_nan += 1

	if data_dict[person]['exercised_stock_options'] == 'NaN':
		exercised_stock_options_nan += 1

	if data_dict[person]['total_stock_value'] == 'NaN':
		total_stock_value_nan += 1

	if data_dict[person]['bonus'] == 'NaN':
		bonus_nan += 1

	if data_dict[person]['salary'] == 'NaN':
		salary_nan += 1



print "poi:", poi_nan
print "exercised_stock_options:", exercised_stock_options_nan
print "total_stock_value:", total_stock_value_nan
print "bonus:", bonus_nan
print "salary:", salary_nan
print "\n"



my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Feature Selection using selectKBest
print "Feature Selection with SelectKBest:"
print "--------------------------------------------------------------- "

# Create dict of feature scores
feature_scores = {}
kbest = SelectKBest(k='all')
selector = kbest.fit_transform(features, labels)

for i, j in zip(kbest.get_support(indices=True), kbest.scores_):
	feature_scores[features_list[i+1]] = j

#print feature_scores
# Create a representation of 'feature_scores' sorted by value in descending order
import operator
sorted_feature_scores = sorted(feature_scores.items(), key=operator.itemgetter(1), reverse = True)
print sorted_feature_scores
print "\n"


### Visualize data
for point in data:
	plt.scatter(point[0], point[2])

#plt.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
print "Classifiers without Validation:" 
print "------------------------------------------------------------------------"

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

clf = GaussianNB()
clf.fit(features, labels)
pred = clf.predict(features)
acc = accuracy_score(pred, labels)
print "GaussianNB accuracy:", acc
print "Precision:", precision_score(y_true=labels, y_pred=pred)
print "Recall:", recall_score(y_true=labels, y_pred=pred)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
pred = clf.predict(features)
acc = accuracy_score(pred, labels)
print "Decision Tree overfitting accuracy:", acc
print "Precision:", precision_score(y_true=labels, y_pred=pred)
print "Recall:", recall_score(y_true=labels, y_pred=pred)

clf = svm.SVC()
clf.fit(features, labels)
pred = clf.predict(features)
acc = accuracy_score(pred, labels)
print "SVC accuracy:", acc
print "Precision:", precision_score(y_true=labels, y_pred=pred)
print "Recall:", recall_score(y_true=labels, y_pred=pred)

clf = neighbors.KNeighborsClassifier()
clf.fit(features, labels)
pred = clf.predict(features)
acc = accuracy_score(pred, labels)
print "KNeighborsClassifier accuracy:", acc
print "Precision:", precision_score(y_true=labels, y_pred=pred)
print "Recall:", recall_score(y_true=labels, y_pred=pred)

clf = AdaBoostClassifier()
clf.fit(features, labels)
pred = clf.predict(features)
acc = accuracy_score(pred, labels)
print "AdaBoostClassifier accuracy:", acc 
print "Precision:", precision_score(y_true=labels, y_pred=pred)
print "Recall:", recall_score(y_true=labels, y_pred=pred)

print "\n"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "Parameter tuning with GridSearchCV:"
print "-------------------------------------------------------------------------"

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV

### GridSearchCV can handle cross validation when the fit function is called
### This means the whole dataset can be passed to the algorithm without partitioning.

### DECISION TREE TUNING
parameters = {'random_state': [None, 40, 45, 50, 55], 'splitter': ("best", "random")}
dt = tree.DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters)
clf.fit(features, labels)
pred = clf.predict(features)
best_clf = clf.best_estimator_
acc = accuracy_score(pred, labels)
print "DecisionTreeClassifier accuracy:", acc
print "Best Parameters:", clf.best_params_

print "Precision:", precision_score(y_true=labels, y_pred=pred)
print "Recall:", recall_score(y_true=labels, y_pred=pred)


print "\n"

print "Results from tester.py:"
print "-------------------------------------------------------------------------"

from tester import test_classifier
test_classifier(best_clf, my_dataset, features_list, folds = 1000)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)