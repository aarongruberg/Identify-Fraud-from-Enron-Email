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

### restricted stock is stock that is given from employer
### to employee and cannot be transferred.
### Stock deferrals delay delivery of shares to employee 
### until a specified date i.e. retirement.
### deferral payments refer to buying something without
### making payments until a specified date.
### deferred income in money that is received upfront 
### but reported in installments i.e. annual fee 12,000
### is reported as 1,000 each month.

### Created new features and excluded any original features
### that contained redundant information such as 'from_this_person_to_poi'
### or 'from_messages'.

### COMPLETE SCALED FEATURES LIST.  Remember that 'poi' is a label.
# features_list = ['poi','salary', 'deferred_income', 'loan_advances', \
#  'other', 'long_term_incentive', 'percent_exercised_stock', \
#  'percent_restricted_stock', 'percent_restricted_stock_deferred', \
#  'percent_to_poi', 'percent_from_poi', 'percent_shared_with_poi', \
#  'percent_deferral_payments', 'percent_expenses', 'percent_director_fees', \
#  'percent_bonus'] # You will need to use more features

### SELECTED FEATURES LIST
### Univariate feature selection with selectKBest
features_list = ['poi', 'salary', 'deferred_income', \
'percent_to_poi', 'percent_bonus']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

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
	
	# new feature 'percent_restricted_stock'
	num = data_dict[person]['restricted_stock']
	den = data_dict[person]['total_stock_value']
	data_dict[person]['percent_restricted_stock'] = computePercent(num, den)
	
	# new feature 'percent_restricted_stock_deferred'
	num = data_dict[person]['restricted_stock_deferred']
	den = data_dict[person]['total_stock_value']
	data_dict[person]['percent_restricted_stock_deferred'] = computePercent(num, den)
	
	# new feature 'percent_to_poi'
	num = data_dict[person]['from_this_person_to_poi']
	den = data_dict[person]['from_messages']
	data_dict[person]['percent_to_poi'] = computePercent(num, den)

	# new feature 'percent_from_poi'
	num = data_dict[person]['from_poi_to_this_person']
	den = data_dict[person]['to_messages']
	data_dict[person]['percent_from_poi'] = computePercent(num, den)

	# new feature 'percent_shared_with_poi'
	num = data_dict[person]['shared_receipt_with_poi']
	den = data_dict[person]['from_messages']
	den += data_dict[person]['to_messages']
	data_dict[person]['percent_shared_with_poi'] = computePercent(num, den)

	# new feature 'percent_deferral_payments'
	num = data_dict[person]['deferral_payments']
	den = data_dict[person]['total_payments']
	data_dict[person]['percent_deferral_payments'] = computePercent(num, den)

	# new feature 'percent_expenses'
	num = data_dict[person]['expenses']
	den = data_dict[person]['total_payments']
	data_dict[person]['percent_expenses'] = computePercent(num, den)

	# new feature 'percent_director_fees'
	num = data_dict[person]['director_fees']
	den = data_dict[person]['total_payments']
	data_dict[person]['percent_director_fees'] = computePercent(num, den)

	# new feature 'percent_bonus'
	num = data_dict[person]['bonus']
	den = data_dict[person]['salary']
	data_dict[person]['percent_bonus'] = computePercent(num, den)


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Feature selection using selectKBest
### Features and scores are printed
kbest = SelectKBest(k=4)
selector = kbest.fit_transform(features, labels)
print "4 best features:"
print "--------------------------------------------------------------- "

for i, j in zip(kbest.get_support(indices=True), kbest.scores_):
	print features_list[i+1], j 


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
print "Classifiers without train/test sets:" 
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


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV

print "After training and testing:" 
print "------------------------------------------------------------------------"

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "GaussianNB accuracy:", acc
print "Precision:", precision_score(y_true=labels_test, y_pred=pred)
print "Recall:", recall_score(y_true=labels_test, y_pred=pred)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "Decision Tree accuracy:", acc
print "Precision:", precision_score(y_true=labels_test, y_pred=pred)
print "Recall:", recall_score(y_true=labels_test, y_pred=pred)

clf = svm.SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "SVC accuracy:", acc
print "Precision:", precision_score(y_true=labels_test, y_pred=pred)
print "Recall:", recall_score(y_true=labels_test, y_pred=pred)

clf = neighbors.KNeighborsClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "KNeighborsClassifier accuracy:", acc
print "Precision:", precision_score(y_true=labels_test, y_pred=pred)
print "Recall:", recall_score(y_true=labels_test, y_pred=pred)

clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "AdaBoostClassifier accuracy:", acc 
print "Precision:", precision_score(y_true=labels_test, y_pred=pred)
print "Recall:", recall_score(y_true=labels_test, y_pred=pred)



print "Parameter tuning with GridSearchCV:"
print "-------------------------------------------------------------------------"


parameters = {'n_estimators': [18, 20, 22, 24], 'learning_rate': [0.10, 0.25, 0.50, 0.75]}
adb = AdaBoostClassifier()
clf = GridSearchCV(adb, parameters)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "AdaBoostClassifier accuracy:", acc
print "Best Parameters:", clf.best_params_

print "Precision:", precision_score(y_true=labels_test, y_pred=pred)
print "Recall:", recall_score(y_true=labels_test, y_pred=pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)