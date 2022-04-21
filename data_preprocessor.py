import pandas as pd
import numpy as np
import os
import pickle
import math
from datetime import timedelta
import random

# constants 
min_seq_len = 3
min_seq_num = 3
min_short_term_len = 5
min_long_term_count = 2
pre_seq_window = 7
random_seed = 2021

# split sequence
def generate_sequence(input_data, min_seq_len, min_seq_num):
	"""Split and filter action sequences for each user

	Args:
		input_data (DataFrame): raw data read from csv file
		min_seq_len (int): minimum length for a sequence to be considered valid
		min_seq_num (int): minumum no. sequences for a user to be considered valid

	Returns:
		total_sequences_dict ({user_id: [[visit_id]]}): daily action sequences for each user
		total_sequences_meta ([(user)[(timestamp(int),seq_len(int))]]) : date and length for each sequnece
	"""

	def _remove_consecutive_visit(visit_record):
		"""remove duplicated consecutive visits in a sequence

		Args:
			visit_record (DataFrame): raw sequences

		Returns:
			clean_sequence (list): sequences with no duplicated consecutive visits
		"""
		clean_sequence = []
		for index, _ in visit_record.iterrows():
			clean_sequence.append(index)
		return clean_sequence

	total_sequences_dict = {} #records visit id in each sequence
	total_sequences_meta = [] # records sequence date and length

	seq_count = 0 # for statistics only

	input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_Time_True'])

	for user in input_data['UserId'].unique(): # process sequences for each user
		user_visits = input_data[input_data['UserId'] == user]
		user_sequences, user_sequences_meta = [], []
		unique_date_group = user_visits.groupby([user_visits['Local_sg_time'].dt.date])
		for date in unique_date_group.groups: # process sequences on each day
			single_date_visit = unique_date_group.get_group(date)
			single_sequence = _remove_consecutive_visit(single_date_visit) 
			if len(single_sequence) >= min_seq_len: # filter sequences too short
				user_sequences.append(single_sequence)
				user_sequences_meta.append((date, len(single_sequence)))
				seq_count += 1
		if len(user_sequences) >= min_seq_num: # filter users with too few visits
			total_sequences_dict[user] = np.array(user_sequences,dtype=object)
			total_sequences_meta.append(user_sequences_meta)
	print(f"Generated {seq_count} sequences in total for {len(total_sequences_dict.keys())} users")
	return total_sequences_dict, total_sequences_meta

# generate sequences of different features
def _reIndex_3d_list(input_list):
	"""Reindex the elements in sequences

	Args:
		input_list (nd_array: [(all users)[(user list)[(seq list)]]]): a 3d list containing all sequences fofr all users

	Returns:
		reIndexed_list (3d list): reindexed list
		index_map ([id]): each element is an original id, the index of an element is the new index the id
	"""
	
	def _flatten_3d_list(input_list):
		"""flattern a 3d list to 1d

		Args:
			input_list (nd_array: [(all users)[(user list)[(seq list)]]]): a 3d list containing all sequences fofr all users

		Returns:
			1d-list: flattened list
		"""
		twoD_lists = input_list.flatten()
		return np.hstack([np.hstack(twoD_list) for twoD_list in twoD_lists])

	def _old_id_to_new(mapping, old_id):
		"""convert old_id to new index by mapping given

		Args:
			mapping ([id]): each element is an original id, the index of an element is the new index the id
			old_id (Any): the original token/id in the list

		Returns:
			int: new index of the token
		"""
		return np.where(mapping == old_id)[0].flat[0]

	flat_list = _flatten_3d_list(input_list) # make 3d list 1d 
	index_map = np.unique(flat_list) # get 
	reIndexed_list = [] 
	for user_seq in input_list: # seq list for each user
		reIndexed_user_list = []
		for seq in user_seq: # each seq
			reIndexed_user_list.append([_old_id_to_new(index_map, poi) for poi in seq])	
		reIndexed_list.append(reIndexed_user_list)
	reIndexed_list = np.array(reIndexed_list, dtype=object)
	
	return reIndexed_list, index_map

def generate_POI_sequences(input_data, visit_sequence_dict):
	"""generate location transition sequences

	Args:
		input_data (DataFrame): raw check-in data
		visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

	Returns:
		reIndexed_POI_sequences (nd_array: [[[POI_index]]]): daily location transition sequences for each user
		POI_reIndex_mapping ([POI_id]): index is the new POI index and element is the original POI id
	"""
	POI_sequences = []

	for user in visit_sequence_dict:
		user_POI_sequences = []
		for seq in visit_sequence_dict[user]:
			single_POI_sequence = []
			for visit in seq:
				single_POI_sequence.append(input_data['VenueId'][visit])
			user_POI_sequences.append(single_POI_sequence)
		POI_sequences.append(user_POI_sequences)
	reIndexed_POI_sequences, POI_reIndex_mapping = _reIndex_3d_list(np.array(POI_sequences,dtype=object))
	return reIndexed_POI_sequences, POI_reIndex_mapping

def generate_category_sequences(input_data, visit_sequence_dict):
	"""generate category transition sequences

	Args:
		input_data (DataFrame): raw check-in data
		visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

	Returns:
		reIndexed_cat_sequences (nd_array: [[[cat_index]]]): daily category transition sequences for each user
		cat_reIndex_mapping ([cat_name]): index is the new category index and element is the original category name
	"""
	cat_sequences = []
	for user in visit_sequence_dict:
		user_cat_sequences = []
		for seq in visit_sequence_dict[user]:
			single_cat_sequence = []
			for visit in seq:
				single_cat_sequence.append(input_data['L1_Category'][visit])
			user_cat_sequences.append(single_cat_sequence)
		cat_sequences.append(user_cat_sequences)
	reIndexed_cat_sequences, cat_reIndex_mapping = _reIndex_3d_list(np.array(cat_sequences,dtype=object))
	return reIndexed_cat_sequences, cat_reIndex_mapping

def generate_user_sequences(input_data, visit_sequence_dict):
	"""generate time (in hour) transition sequences

	Args:
		input_data (DataFrame): raw check-in data
		visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

	Returns:
		reIndexed_user_sequences (nd_array: [[[user_index]]]): daily user sequences (same for each sequence)
		user_reIndex_mapping ([user_id]): index is the new user index and element is the original user id
	"""
	all_user_sequences = []
	for user in visit_sequence_dict:
		user_sequences = []
		for seq in visit_sequence_dict[user]:
			single_user_sequence = [user] * len(seq)
			user_sequences.append(single_user_sequence)
		all_user_sequences.append(user_sequences)
	reIndexed_user_sequences, user_reIndex_mapping = _reIndex_3d_list(np.array(all_user_sequences,dtype=object))
	return reIndexed_user_sequences, user_reIndex_mapping

def generate_hour_sequences(input_data, visit_sequence_dict):
	"""generate time (in hour) transition sequences

	Args:
		input_data (DataFrame): raw check-in data
		visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

	Returns:
		reIndexed_hour_sequences (nd_array: [[[time_index]]]): daily hour transition sequences for each user
		hour_reIndex_mapping ([hour]): index is the new hour index and element is the original hour
	"""
	input_data["hour"] = pd.to_datetime(input_data['Local_Time_True']).dt.hour # add hour column in raw data

	hour_sequences = []
	for user in visit_sequence_dict:
		user_hour_sequences = []
		for seq in visit_sequence_dict[user]:
			single_hour_sequence = []
			for visit in seq:
				single_hour_sequence.append(input_data['hour'][visit])
			user_hour_sequences.append(single_hour_sequence)
		hour_sequences.append(user_hour_sequences)
	reIndexed_hour_sequences, hour_reIndex_mapping = _reIndex_3d_list(np.array(hour_sequences,dtype=object))
	return reIndexed_hour_sequences, hour_reIndex_mapping

def generate_day_sequences(input_data, visit_sequence_dict):
	"""generate weekday/weekend tag for each sequence

	Args:
		input_data (DataFrame): raw check-in data
		visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

	Returns:
		reIndexed_day_sequences (nd_array: [[[day_index]]]): daily weekday/weekend sequences (same for each sequence)
		day_reIndex_mapping ([weekday/weekend]): index is the new day index and element is the original weekday(False)/weekend(True) tag
	"""
	input_data["is_weekend"] = pd.to_datetime(input_data['Local_Time_True']).dt.dayofweek > 4 # add hour column in raw data

	day_sequences = []
	for user in visit_sequence_dict:
		user_day_sequences = []
		for seq in visit_sequence_dict[user]:
			single_day_sequence = []
			for visit in seq:
				single_day_sequence.append(input_data['is_weekend'][visit])
			user_day_sequences.append(single_day_sequence)
		day_sequences.append(user_day_sequences)
	reIndexed_day_sequences, day_reIndex_mapping = _reIndex_3d_list(np.array(day_sequences,dtype=object))
	return reIndexed_day_sequences, day_reIndex_mapping

def generate_dist_matrix_sequences(input_data, visit_sequence_dict):
	"""generate distance matrix for sequences
		e.g., for a sequence [1,2,3], the dist matrix sequences would be:
			[[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]]
			TODO: in model: normalize distance to limit its influence 倒数或固定比例缩放

	Args:
		input_data (DataFrame): raw check-in data
		visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

	Returns:
		dist_matrices (nd_array: [[dist_matrix]]): dist matrix for each daily sequences for each user
	"""

	def _get_distance(pos1, pos2):
		"""Calculate the between two  positions

		Args:
			pos1 ((lat, lon)): coordinates for the position 1
			pos2 ((lat, lon)): coordinates for the position 2

		Returns:
			h_dist (float): distances between two positions
		"""
		lat1, lon1 = pos1
		lat2, lon2 = pos2
		
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		
		a = math.sin(math.radians(dlat / 2)) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(dlon / 2)) ** 2
		c = 2 * math.asin(math.sqrt(a))
		r = 6371 
		h_dist = c * r
		
		return h_dist

	def _generate_dist_matrix(seq, input_data):
		"""Generate a distance matrix for one sequence

		Args:
			seq ([visit_id]): one visit sequence
			input_data (DataFrame): raw check-in data

		Returns:
			dist_matrix ([[d11,d12,...],[d21,d22,...],...]): a matrix show distance between each pair of POI in the sequence
		"""
		return [[_get_distance((input_data['Latitude'][x],input_data['Longitude'][x]),(input_data['Latitude'][y],input_data['Longitude'][y])) \
			 for x in seq] for y in seq]

	dist_matrices = []
	for user in visit_sequence_dict:
		user_dist_matrices = []	
		for seq in visit_sequence_dict[user]: #generate dist matrix for each seq
			dist_matrix = _generate_dist_matrix(seq, input_data)
			user_dist_matrices.append(dist_matrix)
		dist_matrices.append(user_dist_matrices)
	return np.array(dist_matrices, dtype=object)

# generate (short term + long term) feed data

def filter_long_short_term_sequences(total_sequences_meta, min_short_term_len, pre_seq_window, min_long_term_count):
	"""filter valid long+short-term sequences for generation of input data
		criteria: 1. the feed data is composed of multiple long-term sequences and one short-term sequence;
				  2. the short term sequence length >= min_short_term_len(5)
				  3. the long term sequences a sequences 7 days before the short term sequence
				  4. the number of long term sequences must >= min_long_term_count(2)

	Args:
		total_sequences_meta ([(user)[(timestamp(int),seq_len(int))]]): date and length for each sequnece
		min_short_term_len (int): minimum visits in a short-term sequence
		pre_seq_window (int): number of days to look for long-term sequences
		min_long_term_count (int): minimum number of long-term sequences to make the long-short sequences valid

	Returns:
		valid_input_index ([(all users)[(each user)[(valid sequences)seq_index]]]): valid long+short term sequneces for each user
	"""

	valid_input_index = [] # filtered long+short term data
	valid_user_count, valid_input_count = 0, 0, # for statistics purpose

	for _,user_seqences in enumerate(total_sequences_meta): # for each user
		user_valid_input_index = []
		# print(user_seqences)
		for seq_index, seq in enumerate(user_seqences): # for each sequence
			# print(seq)
			# print(seq_index)
			seq_time, seq_len = seq[0], seq[1]
			if seq_len >= min_short_term_len: # valid short-term sequence
				start_time, end_time = seq_time-timedelta(days=pre_seq_window), seq_time
				long_term_seqs = [(index,seq) for index,seq in enumerate(user_seqences[:seq_index]) if seq[0]>=start_time and seq[0]<=end_time]
				if len(long_term_seqs) >= min_long_term_count: # valid long-short term sequence
					user_valid_input_index.append([seq[0] for seq in long_term_seqs] + [seq_index])
					valid_input_count += 1
		valid_input_index.append(user_valid_input_index)
		valid_user_count += 1 if len(user_valid_input_index)>0 else 0
	
	print(f"Filtered {valid_input_count} valid input long+short term sequences for {valid_user_count} users.")
	return valid_input_index

def generate_input_samples(feature_sequences, valid_input_index):
	"""turn a feature sequence into a input long+short term data to be fed into model

	Args:
		feature_sequences (nd_array: [[[id]]]): daily transition sequences for each user for certain feature
		valid_input_index ([(all users)[(each user)[(valid sequences)seq_index]]]): valid long+short term sequneces for each user
	
	Return:
		input_samples ([(input sample)[(sequences)feature_id]]]): valid long+short term featuren sequneces for each user
	"""
	input_samples = []
	for user_index, user_sequences in enumerate(valid_input_index):
		if len(user_sequences)!=0:
			for seq in user_sequences:
				feature_sequence = [feature_sequences[user_index][index] for index in seq]
				input_samples.append(feature_sequence)
	return input_samples

# split train test

def split_train_test(input_samples):
	"""split a input sequence into training, validation and testing sequences
		criteria: train-80%, validation-10%, test-10% 

	Args:
		input_samples (3d array: [(each sample)[(valid sequences)feature_id]]): valid long+short term featuren sequneces for each user

	Returns:
		all_training_samples: 80% of samples for training
		all_validation_samples: 10% of samples for validation
		all_testing_samples: 10% of samples for testing
		all_training_validation_samples: 90% of samples for final training after validation
	"""
	random.Random(random_seed).shuffle(input_samples)
	N = len(input_samples)
	train_valid_boundary = int(0.8*N)
	valid_test_boundary = int(0.9*N)
	all_training_samples = input_samples[:train_valid_boundary]
	all_validation_samples = input_samples[train_valid_boundary:valid_test_boundary]
	all_testing_samples = input_samples[valid_test_boundary:]
	all_training_validation_samples = input_samples[:valid_test_boundary]

	return all_training_samples,all_validation_samples,all_testing_samples,all_training_validation_samples

def reshape_data(original_data):
	"""combine different samples for each features to one sample containing all features

	Args:
		original_data ([features * sample * sequence]): combination of samples for each feature

	Return:
		reshaped_data ([sample * sequence * features]): each sample contains myltiple features
	"""
	result = []

	# [feature * sample] -> [sample * feature]
	samples = np.transpose(np.array(original_data, dtype=object), (1,0))

	# [feature * sequence] -> [sequence * feature]

	for sample in samples:
		sample_data = []
		feature_num = len(sample) # 6 features
		sequence_num = len(sample[0]) # number of steps in this sequence
		for i in range(sequence_num):
			sample_data.append([sample[j][i] for j in range(feature_num)])
		result.append(sample_data)
	return result


def dump_data(data, city, data_type):
	"""save data as pickle file

	Args:
		data ([(feature)[(sample)[feature_id]]]): processed data
		city (str): city code for file naming
		data_type (str): data description for file naming
	"""
	directory = './processed_data'
	if not os.path.exists(directory):
		os.makedirs(directory)
	file_path = directory + "/{}_{}"

	pickle.dump(data, open(file_path.format(city,data_type), 'wb'))

# completely process data for one city

def generate_data(city):
	"""Generate complete train and test data set for one city
		Save the result in pickle files

	Args:
		city (str): city to read data from and process 
	"""
	print(f"******Process data for {city}******")

	if city == "Singapore":
		city_code = "SIN"
	elif city == "New York":
		city_code = "NY" 
	elif city == "Pheonix":
		city_code = "PHO"
	else:
		raise "City name not supported"

	data = pd.read_csv(f"./raw_data/{city_code}_checkin_with_active_regionId.csv")
	
	visit_sequence_dict, total_sequences_meta = generate_sequence(data, min_seq_len, min_seq_num)
	valid_input_index = filter_long_short_term_sequences(total_sequences_meta, min_short_term_len, pre_seq_window, min_long_term_count)

	train_data, valid_data, test_data, train_valid_data, meta_data = [], [], [], [], {}

	# poi inputs
	poi_sequences, poi_mapping = generate_POI_sequences(data, visit_sequence_dict)
	poi_input_data = generate_input_samples(poi_sequences, valid_input_index)
	poi_train, poi_valid, poi_test, poi_train_valid = split_train_test(poi_input_data)
	train_data.append(poi_train)
	valid_data.append(poi_valid)
	test_data.append(poi_test)
	train_valid_data.append(poi_train_valid)
	print("poi sequence generated.")

	# cat inputs
	cat_sequences, cat_mapping= generate_category_sequences(data, visit_sequence_dict)
	cat_input_data = generate_input_samples(cat_sequences, valid_input_index)
	cat_train, cat_valid, cat_test, cat_train_valid = split_train_test(cat_input_data)
	train_data.append(cat_train)
	valid_data.append(cat_valid)
	test_data.append(cat_test)
	train_valid_data.append(cat_train_valid)
	print("category sequence generated.")

	# user inputs
	user_sequences, user_mapping = generate_user_sequences(data, visit_sequence_dict)
	user_input_data = generate_input_samples(user_sequences, valid_input_index)
	user_train, user_valid, user_test, user_train_valid = split_train_test(user_input_data)
	train_data.append(user_train)
	valid_data.append(user_valid)
	test_data.append(user_test)
	train_valid_data.append(user_train_valid)
	print("user sequence generated.")

	# hour inputs
	hour_sequences, hour_mapping = generate_hour_sequences(data, visit_sequence_dict)
	hour_input_data = generate_input_samples(hour_sequences, valid_input_index)
	hour_train, hour_valid, hour_test, hour_train_valid = split_train_test(hour_input_data)
	train_data.append(hour_train)
	valid_data.append(hour_valid)
	test_data.append(hour_test)
	train_valid_data.append(hour_train_valid)
	print("hour sequence generated.")

	# day inputs
	day_sequences, day_mapping = generate_day_sequences(data, visit_sequence_dict)
	day_input_data = generate_input_samples(day_sequences, valid_input_index)
	day_train, day_valid, day_test, day_train_valid = split_train_test(day_input_data)
	train_data.append(day_train)
	valid_data.append(day_valid)
	test_data.append(day_test)
	train_valid_data.append(day_train_valid)
	print("day sequence generated.")

	# dist inputs
	dist_matrix_sequences = generate_dist_matrix_sequences(data, visit_sequence_dict)
	dist_matrix_input_data = generate_input_samples(dist_matrix_sequences, valid_input_index)
	dist_train, dist_valid, dist_test, dist_train_valid = split_train_test(dist_matrix_input_data)
	train_data.append(dist_train)
	valid_data.append(dist_valid)
	test_data.append(dist_test)
	train_valid_data.append(dist_train_valid)
	print("dist sequence generated.")

	# reshape data: [features * sample * sequence] -> [sample * sequence * features]
	train_data = reshape_data(train_data)
	valid_data = reshape_data(valid_data)
	test_data = reshape_data(test_data)
	train_valid_data = reshape_data(train_valid_data)

	# meta data
	meta_data["POI"] = poi_mapping
	meta_data["cat"] = cat_mapping
	meta_data["user"] = user_mapping
	meta_data["hour"] = hour_mapping
	meta_data["day"] = day_mapping

	# output data
	dump_data(train_data, city_code, "train")
	dump_data(valid_data, city_code, "valid")
	dump_data(test_data, city_code, "test")
	dump_data(train_valid_data, city_code, "train_valid")
	dump_data(meta_data, city_code, "meta")


if __name__ == '__main__':
	city_list = ["Pheonix"]
	
	for city in city_list:
		generate_data(city) 