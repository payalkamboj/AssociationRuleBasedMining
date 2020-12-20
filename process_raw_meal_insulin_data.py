import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import time

#reading cgm data of patient
cgm_data = pd.read_csv("Data/CGMData.csv")

cgm_data = cgm_data.interpolate(method ='linear', limit_area=None)

#cgm_data.dropna()
cgm_data['DateTime'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
cgm_data['GlobalIndex'] = range(0, len(cgm_data))

#reading insulin data
insulin_data = pd.read_csv("Data/InsulinData.csv")
insulin_data['DateTime'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])
insulin_data['GlobalIndex'] = range(0, len(insulin_data))

insulin_data = insulin_data.fillna(0)

def extract_inbetween_meal_data(insulin_dataframe, index_begin):
	t_begin_max = insulin_dataframe['DateTime'].iloc[0] - timedelta(minutes=120)
	t_begin = insulin_dataframe.loc[insulin_dataframe.GlobalIndex == index_begin, 'DateTime'].item()

	if (t_begin > t_begin_max):
		return [None, None]
	
	global_index = index_begin
	t_end_loop = t_begin + timedelta(minutes=120)

	# As index_begin is the 'tm' meal point, check for all the points in the next two hour range
	# starting with next index
	global_index -= 1
	t_begin_loop = insulin_dataframe.loc[insulin_dataframe.GlobalIndex == global_index, 'DateTime'].item()
	
	while (t_begin_loop <= t_end_loop):
		carbs = insulin_dataframe.loc[insulin_dataframe.GlobalIndex == (global_index), 'BWZ Carb Input (grams)'].item()
		
		if (carbs > 0.001):
			#print(carbs)
			return extract_inbetween_meal_data(insulin_dataframe, global_index)
		
		# Decrementing index for next available data/timestamp
		global_index -= 1
		# Finding the DateTime corresponsing to next available data/timestamp
		t_begin_loop = insulin_dataframe.loc[insulin_dataframe.GlobalIndex == global_index, 'DateTime'].item()
	
	bolus_insulin_at_meal_time = round(insulin_dataframe.loc[insulin_dataframe.GlobalIndex == index_begin, 'BWZ Estimate (U)'].item())	
	return ([t_begin - timedelta(minutes=30), t_begin + timedelta(minutes=120)], global_index, bolus_insulin_at_meal_time)

# Extracting meal data datetime from insulin data
def extract_meal_data_insulin(insulin_dataframe):
	meal_data = []
	num_unique_indices = insulin_dataframe['GlobalIndex'].nunique()

	global_index = insulin_dataframe['GlobalIndex'].iloc[-1]

	# Datetime where the test data begins
	t_begin = insulin_dataframe.loc[insulin_dataframe.GlobalIndex == global_index, 'DateTime'].item()
	# Last datetime beyond which enough data is not available to be considered either/meal or no-meal (need atleast 2 hours)
	#t_begin_max = insulin_dataframe['DateTime'].iloc[0] - timedelta(minutes=120)
	t_begin_max = insulin_dataframe['DateTime'].iloc[0] - timedelta(hours=2)

	while (t_begin <= t_begin_max):
		carb_value = insulin_dataframe.loc[insulin_dataframe.GlobalIndex == global_index, 'BWZ Carb Input (grams)'].item()
		
		# Keeping in mind the floating point (im)precision
		if (carb_value > 0.001):
			# Found a non-Nan carb entry, now extract a continuous [tm - 30, tm + 120] time interval
			temp_meal_data = extract_inbetween_meal_data(insulin_dataframe, global_index)
			# Update the global index to point at an index right after the current index position to ensure 
			# next iteration starts where the current iteration ended 
			global_index = temp_meal_data[1] - 1
			temp_meal_data = [temp_meal_data[0][0],temp_meal_data[0][1],temp_meal_data[2]]
			meal_data.append(temp_meal_data)
			t_begin = insulin_dataframe.loc[insulin_dataframe.GlobalIndex == global_index, 'DateTime'].item()
			#print('Meal_Data:',meal_data)
			continue

		# Keep moving forward one step at a time untill we see a 'non-Nan' carb entry
		global_index -= 1
		t_begin = insulin_dataframe.loc[insulin_dataframe.GlobalIndex == global_index, 'DateTime'].item()

	return meal_data

def slice_data_frame(unsliced_data, datetime_begin, datetime_end):

	sliced_data = unsliced_data[(unsliced_data['DateTime'] >= datetime_begin) & (unsliced_data['DateTime'] <= datetime_end)]

	return sliced_data

def get_cgm_insulin_meal_data(meal_list, cgm_data):
	
	len_meal_data = len(meal_list)
	
	# Initializing empty numpy array of appropriate dimensions for meal data
	cgm_meal_data_max = np.empty((0,1))
	cgm_meal_data_min = np.empty((0,1))
	meal_time_insulin_data = np.empty((0,1))
	cgm_meal_data_at_mealtime = np.empty((0,1))
	
	for index in range(len_meal_data):
		t_begin = meal_list[index][0]
		t_end = meal_list[index][1]

		# Extracting the whole time series strech for this meal
		temp_cgm_meal_data = slice_data_frame(cgm_data,t_begin,t_end)
		temp_cgm_meal_glucose_data = temp_cgm_meal_data['Sensor Glucose (mg/dL)']
		temp_cgm_meal_glucose_data.dropna()
		if (len(temp_cgm_meal_glucose_data) < 30):
			continue

		# Extracting the CGM value corresponding to this meal-time, B_Meal (at time >= mealtime)
		temp_cgm_meal_data_at_mealtime = slice_data_frame(cgm_data,t_begin + timedelta(minutes=30),t_end)
		# Take the first instance of glucose reading in this sliced data being the first record >= mealtime
		# Note that data is in reverse chronological order, so last element is the earliest instance
		temp_cgm_meal_data_at_mealtime = temp_cgm_meal_data_at_mealtime['Sensor Glucose (mg/dL)']
		temp_cgm_meal_data_at_mealtime = temp_cgm_meal_data_at_mealtime.iloc[-1]
		cgm_meal_data_at_mealtime = np.append(cgm_meal_data_at_mealtime,temp_cgm_meal_data_at_mealtime)

		# Protecting for un-intended longer than expected time-series data from CGM
		temp_cgm_meal_glucose_data = temp_cgm_meal_glucose_data[:30]
		temp_cgm_meal_glucose_data = temp_cgm_meal_glucose_data.T
		temp_cgm_meal_glucose_data = temp_cgm_meal_glucose_data.to_numpy()
		cgm_meal_data_max = np.append(cgm_meal_data_max,np.amax(temp_cgm_meal_glucose_data))
		cgm_meal_data_min = np.append(cgm_meal_data_min,np.amin(temp_cgm_meal_glucose_data))
		meal_time_insulin_data = np.append(meal_time_insulin_data,meal_list[index][2])

	df_cgm_meal_data_max = pd.DataFrame(data = cgm_meal_data_max, columns = ['CGM_Glucose_Max'])
	df_cgm_meal_data_min = pd.DataFrame(data = cgm_meal_data_min, columns = ['CGM_Glucose_Min'])
	df_meal_time_insulin_data = pd.DataFrame(data = meal_time_insulin_data, columns = ['Bolus_Insulin_at_Mealtime'],  dtype='int32')
	df_cgm_meal_data_at_mealtime = pd.DataFrame(data = cgm_meal_data_at_mealtime, columns = ['CGM_Glucose_at_Mealtime'])
	df_cgm_meal_insulin_data = pd.concat([df_cgm_meal_data_max,df_cgm_meal_data_min,df_meal_time_insulin_data,df_cgm_meal_data_at_mealtime], axis=1)

	return df_cgm_meal_insulin_data

if __name__ == "__main__":

	t = time.time()

	debug = False
	usePickledCGMRawData = False

	if usePickledCGMRawData == False:
		meal_list = extract_meal_data_insulin(insulin_data)
		cgm_insulin_raw_meal_data = get_cgm_insulin_meal_data(meal_list, cgm_data)
		print('cgm_insulin_raw_meal_data: \n',cgm_insulin_raw_meal_data)
		pickle.dump(cgm_insulin_raw_meal_data, open("Data/cgm_insulin_raw_meal_data.p", "wb" ) )
	else:
		cgm_insulin_raw_meal_data = pickle.load(open("Data/cgm_insulin_raw_meal_data.p", "rb" ) )

	elapsed = time.time() - t
	print('Elapsed Time:',elapsed)