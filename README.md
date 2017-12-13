### Workflow

<p>The goal of this project was to write a script 'poi_id.py' to predict an employee's poi label based on their financial and email features.  The script 'helper.py' contains helper functions for finding the max/min values of a feature and creating new features based on percentages.</p>

<p>'poi_id.py' opens the 'final_project_dataset.pkl' file and sorts the data into a dictionary of dictionaries.  The 'feature_format.py' file was imported to extract the features and labels from the dataset.  Multiple algorithms were deployed.  One was selected and tuned for final analysis.  The 'tester.py' file was imported to evaluate the performance of the final algorithm and write the pickle files for the classifier, dataset, and features.</p>

### Identify Fraud from Enron Email

<p>Machine learning techniques were applied to data from 146 Enron employees.  There were 21 features for each employee and 3066 total data points.  The data for each employee fell into 3 categories: financial, email, and poi label.  The poi label signified whether or not an employee was a person of interest in the Enron case.  18 employees were labeled as a poi and 128 employees were not.  5 machine learning algorithms were deployed to predict an employee's poi label based on their financial and email features.  1 algorithm was selected and tuned for final analysis.</p>

### Data Structure

The data was stored in a dictionary of dictionaries.  For the outer dict, each key was a person and each value was a feature name.  For the inner dict, each key was a feature name and each value was the value of that feature.  

```Python
datadict['SKILLING JEFFREY K']['salary']
```

###### Financial Features

<p>['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)</p>

###### Email Features

<p>['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)</p>

###### POI Label

<p>[‘poi’] (boolean, represented as integer)</p>

### Missing Values

<p>These are the total 'NaN' values for each feature.</p>

POI: 0  
Salary: 50  
Deferred Income: 96  
Loan Advances: 141  
Other: 53  
Long Term Incentive: 79  
Percent Exercised Stock: 44  
Percent Restricted Stock: 38  
Percent Restricted Stock Deferred: 130  
Percent to POI: 58  
Percent from POI: 58  
Percent Shared with POI: 58  
Percent Deferral Payments: 106  
Percent Expenses: 50  
Percent Director Fees: 130  
Percent Bonus: 63  

### Outliers

<p>Many outliers in this dataset were important because they helped identify persons of interest.  However, some outliers did not correspond to a person.  The max value for financial features was from a 'TOTAL' key rather than a 'PERSON' key.  Another non-person key 'THE TRAVEL AGENCY IN THE PARK' was identified and these keys were removed from the data dictionary.</p>

### Feature Selection

<p>SelectKBest was used to get the feature importances for all features.  The feature importances are based on the chi squared value between each feature and the poi label.  The four features with the highest importance scores were selected.  The first element 'poi' is a label.  Feature scaling was not required for the decision tree algorithm that was deployed.</p>      

```Python
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus', 'salary']
```  

exercised_stock_options: 24.815079733218194  
total_stock_value: 24.182898678566879  
bonus: 20.792252047181535  
salary: 18.289684043404513  
deferred_income: 11.458476579280369  
long_term_incentive: 9.9221860131898225  
restricted_stock: 9.2128106219771002  
total_payments: 8.7727777300916756  
shared_receipt_with_poi: 8.589420731682381  
loan_advances: 7.1840556582887247  
expenses: 6.0941733106389453  
from_poi_to_this_person: 5.2434497133749582  
other: 4.1874775069953749  
from_this_person_to_poi: 2.3826121082276739  
director_fees: 2.1263278020077054  
to_messages: 1.6463411294420076  
deferral_payments: 0.22461127473600989  
from_messages: 0.16970094762175533  
restricted_stock_deferred: 0.065499652909942141

### Feature Creation

<p>Additional features 'percent_exercised_stock' and 'percent_bonus' were created but their feature importances were very low.  'percent_exercised_stock' was the percent of 'total_stock_value' that was exercised and 'percent_bonus' was the percent of 'salary' equal to an employee's bonus.</p>

exercised_stock_options: 21.153646538437151  
total_stock_value: 20.492888346982209  
bonus: 17.326074648455403  
salary: 14.579307471130718  
percent_bonus: 8.5695799793786058  
percent_exercised_stock: 0.6724199516795144  

### Fitting classifiers without validation

<p>A variety of classifiers were fit to all of the features and labels in the data set.  The features were used to make predictions about labels and these predicted labels were compared with the true labels.  The accuracy, precision, and recall of each algorithm was measured.</p>  

<p>The precision is the amount of true positives divided by the sum of true positives and false positives.  It is the (correctly identified pois)/(correctly identified pois + incorrectly identified pois).  Precision is a measure of exactness.  The recall is the (correctly identified pois)/(correctly identified pois + incorrectly identified non-pois).  Recall is a measure of completeness.  It gives the percentage of total pois in the dataset that were identified.</p>

###### GaussianNB

accuracy: 0.869230769231  
precision: 0.545454545455  
recall: 0.333333333333

###### Decision Tree

accuracy: 1.0  
precision: 1.0  
recall: 1.0  

###### SVC

accuracy: 1.0  
precision: 1.0  
recall: 1.0  

###### KNeighbors

accuracy: 0.907692307692  
precision: 0.8  
recall: 0.444444444444  

###### AdaBoost

accuracy: 1.0  
precision: 1.0  
recall: 1.0 

### GridSearchCV

###### Cross Validation

<p>Validation is important because it allows us to test our algorithms performance and avoid overfitting to the data.</p>

<p>A decision tree algorithm was tuned using GridSearchCV.  Because GridSearchCV can do cross validation, the whole data set was passed to the algorithm instead of splitting the data into testing and training sets.  This is more useful because when data is split into train/test sets, the goal is to maximize the training set size to acheive the best learning outcome as well as maximize the test set size to acheive the best validation.  Partitioning the data like this caused a tradeoff because every data point used for the test set cannot be used by the training set and visa versa.  Cross validation partitions the data into bins of equal size and the learning experiment is run once for every bin in the set.  One bin is selected as the test set and the rest of the bins are selected for training.   This is done for every bin and the test results for each bin are averaged.  This takes more computing time but has better accuracy than training and testing sets.  The same data should not be used for both training and testing.  If a model fits to a training set and then tests on that same data then the model is not being tested or validated.</p>  

###### Parameter Tuning

<p>Poor parameter tuning can decrease the performance of the classifier.  It can also cause the classifier to overfit to the training data making it more difficult to perform well on the test data.  In the decision tree algorithm, the 'max_depth' parameter helps control the size of the tree.  This is important because if the tree gets too big the algorithm will overfit to the training data and have trouble validating on the test data.  The 'min_samples_split' parameter controls the number of samples at each node.  "A very small number will usually mean the tree will overfit, whereas a large number will prevent the tree from learning the data.(sklearn documentation)"  First, 'max_depth' was set to 3 while 'min_samples_split' was varied, then 'min_samples_split' was set to 5 while 'max_depth' was gradually increased.</p>

###### Decision Tree

best_parameters: {'max_depth': 10, 'min_samples_split': 6}  
accuracy: 0.79792  
precision: 0.32708   
recall: 0.29650 

### Effects of Additional Features

<p>The decision tree algorithm was attempted with one of the percentage features created earlier, rather than the original features.  The results were slightly worse.</p>  
```Python
features_list = ['poi','percent_exercised_stock', 'bonus', 'salary']
```

###### Decision Tree

best_parameters: {'max_depth': 10, 'min_samples_split': 5}  
accuracy: 0.78254  
precision: 0.29232    
recall: 0.29100    

### Improving Feature Selection

<p>I tried the decision tree algorithm again with only the top 3 features from the SelectKBest scores earlier.  The results were slightly better.</p>

```Python
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus']
```  
###### Decision Tree

best_parameters: {'max_depth': 10, 'min_samples_split': 6}  
accuracy: 0.81000   
precision: 0.36479    
recall: 0.31700   
