#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt

from outliers import find_max

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
features_list = ['poi','salary', 'from_this_person_to_poi', \
 'from_poi_to_this_person', 'total_stock_value', \
 'deferral_payments', 'restricted_stock_deferred', \
 'deferred_income', 'total_payments', 'loan_advances', \
 'bonus', 'expenses', 'exercised_stock_options', \
 'other', 'long_term_incentive', 'restricted_stock',\
 'director_fees', 'to_messages', 'from_messages',
 'shared_receipt_with_poi', 'percent_exercised_stock', \
 'percent_to_poi', 'percent_from_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# many outliers in this dataset are important and should be kept
# find_max revealed 'TOTAL' to have the max value
# for financial features.  Remove the 'TOTAL' key. 
data_dict.pop('TOTAL', 0)

# find the max/min value and person for a feature
print find_max(data_dict, 'bonus')


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# new feature 'percent_exercised_stock'
for person in data_dict.keys():
	num = float(data_dict[person]['exercised_stock_options'])
	den = float(data_dict[person]['total_stock_value'])
	data_dict[person]['percent_exercised_stock'] = num/den

# new feature 'percent_to_poi'
for person in data_dict.keys():
	num = float(data_dict[person]['from_this_person_to_poi'])
	den = float(data_dict[person]['from_messages'])
	data_dict[person]['percent_to_poi'] = num/den 

# new feature 'percent_from_poi'
for person in data_dict.keys():
	num = float(data_dict[person]['from_poi_to_this_person'])
	den = float(data_dict[person]['to_messages'])
	data_dict[person]['percent_from_poi'] = num/den



my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Visualize data
for point in data:
	poi = point[0] 
	salary = point[1]
	to_poi = point[2]
	from_poi = point[3]
	total_stock = point[4]
	deferral_payments = point[5]
	restricted_stock_deferred = point[6]
	deferred_income = point[7]
	total_payments = point[8]
	loan_advances = point[9]
	bonus = point[10]
	expenses = point[11]
	exercised_stock_options = point[12]
	other = point[13]
	long_term_incentive = point[14]
	restricted_stock = point[15]
	director_fees = point[16]
	to_messages = point[17]
	from_messages = point[18]
	shared_receipt_with_poi = point[19]
	percent_exercised_stock = point[20]
	percent_to_poi = point[21]
	percent_from_poi = point[22]
	plt.scatter(poi, percent_from_poi)

plt.show()


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)