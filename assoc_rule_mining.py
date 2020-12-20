
import numpy as np
import pandas as pd
import time
import pickle
import math    

if __name__ == "__main__":

	t = time.time()

	cgm_insulin_raw_meal_data = pickle.load(open("Data/cgm_insulin_raw_meal_data.p", "rb" ) )

	cgm_meal_data_max = cgm_insulin_raw_meal_data['CGM_Glucose_Max']
	cgm_meal_data_min = cgm_insulin_raw_meal_data['CGM_Glucose_Min']
	cgm_meal_data_at_mealtime = cgm_insulin_raw_meal_data['CGM_Glucose_at_Mealtime']
	meal_time_insulin_data = cgm_insulin_raw_meal_data['Bolus_Insulin_at_Mealtime']

	#print(cgm_meal_data_max)
	#print(cgm_meal_data_min)
	#print(cgm_meal_data_at_mealtime)
	#print(meal_time_insulin_data)

	cgm_global_min = cgm_meal_data_min.values.min()
	cgm_global_max = cgm_meal_data_max.values.max()

	print('cgm_global_min',cgm_global_min)
	print('cgm_global_max',cgm_global_max)

	bin_size = 20.0
	num_bins = int((cgm_global_max - cgm_global_min) / bin_size)

	cut_bins = []
	cut_labels = []

	for i in range(num_bins + 1):

		cut_bins.append((cgm_global_min + i*bin_size))

	for i in range(num_bins):

		cut_labels.append("Bin_" + str(i))

	# Binning
	cgm_insulin_raw_meal_data['B_Max'] = pd.cut(cgm_insulin_raw_meal_data['CGM_Glucose_Max'], bins=cut_bins, labels=cut_labels, include_lowest = True)
	cgm_insulin_raw_meal_data['B_Meal'] = pd.cut(cgm_insulin_raw_meal_data['CGM_Glucose_at_Mealtime'], bins=cut_bins, labels=cut_labels, include_lowest = True)
	#cgm_insulin_raw_meal_data.to_csv('cgm_insulin_raw_meal_data.csv')
	n_samples = len(cgm_insulin_raw_meal_data)

	dataMap = {}
	for i in range(n_samples):
		key = (cgm_insulin_raw_meal_data.iloc[i]['B_Max'],cgm_insulin_raw_meal_data.iloc[i]['B_Meal'],cgm_insulin_raw_meal_data.iloc[i]['Bolus_Insulin_at_Mealtime'])
		if key in dataMap.keys():
			dataMap[key] += 1
		else:
			dataMap[key] = 1

# 1. Most Frequent ItemSets (exluding itemsets with only single occurence)
	mostFrequentItemSets = pd.DataFrame(columns = ['ItemSet (B_Max, B_Meal, Insulin Bolus)','Frequency'])
	index = 0
	for w in sorted(dataMap, key=dataMap.get, reverse=True):
		if (dataMap[w] <= 1):
			break
		mostFrequentItemSets.loc[index] = pd.Series([w,dataMap[w]],index= mostFrequentItemSets.columns)
		index += 1
		
	#print(mostFrequentItemSets)
	mostFrequentItemSets.to_csv('Data/mostFrequentItemSets.csv', index  = False)

# 2. Rule(s) with (highest) confidence. 

	antecedentFrequencyMap = {}
	for i in range(n_samples):
		key = (cgm_insulin_raw_meal_data.iloc[i]['B_Max'],cgm_insulin_raw_meal_data.iloc[i]['B_Meal'])
		if key in antecedentFrequencyMap.keys():
			antecedentFrequencyMap[key] += 1
		else:
			antecedentFrequencyMap[key] = 1

	rulesConfidencedataMap = dataMap
	
	for key,value in rulesConfidencedataMap.items():
		rulesConfidencedataMap[key] = rulesConfidencedataMap[key]/antecedentFrequencyMap[key[:-1]]*100

	index = 0
	rulesConfidencedataMapSorted = pd.DataFrame(columns = ['Rule [{B_Max, B_Meal} -> I_B]','Confidence'])

	for w in sorted(rulesConfidencedataMap, key=rulesConfidencedataMap.get, reverse=True):
		rulesConfidencedataMapSorted.loc[index] = pd.Series(["(" + str(w[0]) + "," +  str(w[1]) + ")" + " -> " + str(w[2]),rulesConfidencedataMap[w]],index= rulesConfidencedataMapSorted.columns)
		index += 1

	# Here, itemsets with 100% confidence would naturally win
	# We observe that these 100% confidence itemsets have low support. For instance, a single occurence
	# itemset is resulting in 100% confidence as that {B_Max, B_Meal} combination has only one instance

	max_confidence  = rulesConfidencedataMapSorted.iloc[0]['Confidence']
	
	rulesMaxConfidencedataMap = rulesConfidencedataMapSorted[rulesConfidencedataMapSorted.Confidence >= max_confidence]

	rulesMaxConfidencedataMap.to_csv('Data/rulesMaxConfidencedataMap.csv', index = False)

# 3. Anomalous Rules, with confidence less than 15%
	rulesConfidencedataMapAnomalous = rulesConfidencedataMapSorted[rulesConfidencedataMapSorted.Confidence < 15.0]
	rulesConfidencedataMapAnomalous.to_csv('Data/rulesConfidencedataMapAnomalous.csv', index = False)