#!/usr/bin/python

### finds the name of the person with max value for a feature
def find_max(data_dict, feature):
	feature_list = []
	for person in data_dict.keys():
		if data_dict[person][feature] != 'NaN':
			feature_list.append(data_dict[person][feature])


	for person in data_dict.keys():
		if data_dict[person][feature] == max(feature_list):
			feature_max = 'max:', person, data_dict[person][feature]
		elif data_dict[person][feature] == min(feature_list):
			feature_min = 'min:', person, data_dict[person][feature]

	return feature_max, feature_min

### computes percentage of sub feature contained in a total feature
### creates new feature 'percent of feature'
def computePercent(num, den):
	percent = 0

	if den != 0 and num != 'NaN' and den != 'NaN':
		num = float(num)
		den = float(den)
		percent = num/den
	else:
		percent = 'NaN'
		
	return percent





