#this contains all the utils needed for machine learning projects
import tensorflow as tf 
import numpy as np
import os
import gzip
import pickle
import scipy as scp
import random
import time
import lycon
import h5py
#setup the iterator:
def get_iterator(batch_size, **kwargs):
	"""
	Args:
		batch_size
			- the size to batch the data
		**kwargs
			- the datasets and related indexed names
	"""
	dataset = tf.data.Dataset.from_tensor_slices(kwargs).shuffle(10000).batch(batch_size).repeat()
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()
	return iterator, next_element



#get the dataset
def get_mnist_data(datapath, shuffle=False):
	"""
	Args:
		datapath
			- this is the path for the MNIST dataset.
	Returns:
		training_data
			- training data
			- in the form of a dictionary. "labels": labels, "data":data
		validation_data
			- validation data
			- in the form of a dictionary. "labels": labels, "data":data
		test_data
			- test data
			- in the form of a dictionary. "labels": labels, "data":data
	"""
	data_file = "MNIST/mnist.pkl.gz"
	mnist_path = os.path.join(datapath, data_file)
	with gzip.open(mnist_path, "rb") as f:
		datasets = pickle.load(f, encoding="latin1")
	
	def create_dataset_dict(*args):
		#takes in tuple of data, labels and creates a dictionary.
		new_set = {}
		for i in range(len(args)):
			dataset = args[i]
			data = dataset[0].reshape(-1,28,28, 1)
			if shuffle:
				np.random.shuffle(data)
			labels = dataset[1]
			if new_set == {}:
				new_set["data"] = data
				new_set["labels"] = labels
				new_set["labels_names"] = ["one","two","three","four","five","six","seven","eight","nine","ten"]
			new_set["data"] = np.concatenate((new_set["data"] , data), axis=0)
			new_set["labels"] = np.concatenate((new_set["labels"] , labels), axis=0)
		return new_set
	ret = create_dataset_dict(*datasets)
	return ret["data"], ret["labels"]

def loading_bar(cur, total):
	fraction = cur/total
	string = "[%-20s]\t%.2f%%\t%d/%d\t\t"%("="*int(20*fraction), fraction*100, cur, total)
	return string
"""
def get_celeba_data(datapath, save_new=False):
	#loads from numpy file, if available or specified, else save to numpy file
	datapath = os.path.join(datapath, "celeba")
	labels_file = "list_attr_celeba.txt"
	images_saved_file = "images_saved.hdf5"
	images_saved_path = os.path.join(datapath, images_saved_file)
	labels_path = os.path.join(datapath, labels_file)
	images_path = os.path.join(datapath, "images")

	if (not save_new) and os.path.exists(images_saved_path):
		#load previous
		print("found previous loaded...loading")
		data = np.load(images_saved_path)
		return data["images"], data["labels"]
	else:
		print("previous load not found... loading")
		save_new = True
	
	#get previous saved path.
	with open(labels_path,"r") as f:
		total_labels = f.readlines()

	#get the labels:
	num_images = int(total_labels[0])
	filenames = []
	labels = []
	labels_names = total_labels[1].split()
	print("loading labels...")
	for line in total_labels[2:]:
		labels.append(line.split()[1:])
		filenames.append(os.path.join(images_path, line.split()[0]))
	labels = (np.asarray(labels).astype(int)+1)/2
	
	dataset = get_data()
	dataset.set_labels(labels[:500])
	dataset.get_images_from_filenames(filenames[:500])
	dataset.save_file(images_saved_path)
"""
def get_celeba_data(datapath, save_new=False, get_group=True, group_num=1, shuffle=False, max_len_only=True, is_HD=True, **kwargs):
	"""
	This will retrieve the celeba dataset

	Examples:
		>>> dataset, get_group = gu.get_celeba_data(gc.datapath, group_num=1)
		>>> images_1, labels_1 = get_group()
		>>> images_2, labels_2 = get_group()

	Args:
		datapath:  This is the datapath the the general data folder
		save_new:  saves a hdf5 dataset if true, or not available. uses old one if false
		get_group:  Makes this function return dataset and group objects (for chunked data loading),
			otherwise, load all images and labels. get_group is an iterator which will load data.
		group_num:  The number of groups to load at once
		shuffle:  Shuffles the groups, if loading with groups
		max_len_only:  This will force the groups to be of max length.
		is_HD: Whether to extract the hd version (True) or default version (False)
		**kwargs: These are any other irrelevant kwargs.

	Returns:
		data, labels if not get group, else dataset object, get_group object.
	"""
	#loads from numpy file, if available or specified, else save to numpy file
	if is_HD:
		datapath = os.path.join(datapath, "celeba_HD", "dataset")
	else:
		datapath = os.path.join(datapath, "celeba")
	labels_file = "list_attr_celeba.txt"
	images_saved_file = "images_saved.hdf5"
	images_saved_path = os.path.join(datapath, images_saved_file)
	labels_path = os.path.join(datapath, labels_file)
	images_path = os.path.join(datapath, "images")

	dataset = get_data(images_saved_path)
	if not save_new:
		ret = dataset.possible_load_group_indicies(images_saved_path, shuffle)
		if ret:
			save_new = True

	if save_new:
		if os.path.exists(images_saved_path):
			os.remove(images_saved_path)
		#get the labels and images filenames:
		#get previous saved path.
		with open(labels_path,"r") as f:
			total_labels = f.readlines()

		#get the labels:
		filenames = []
		labels = []
		labels_names = total_labels[1].split()
		print("loading labels...")
		for line in total_labels[2:]:
			labels.append(line.split()[1:])
			filenames.append(os.path.join(images_path, line.split()[0]))
		labels = (np.asarray(labels).astype(int)+1)/2
		
		dataset.save_by_group(labels, filenames, 64)
	
	if get_group:
		#returns dataset_object and the get next method
		dataset.possible_load_group_indicies(shuffle, max_len_only)
		return dataset, lambda group_num=group_num, random_selection=True, remove_past=False: dataset.get_next_group(
								random_selection, group_num, remove_past)
		 
	else:
		dataset.load()

	return dataset.images, dataset.labels

	


class get_data():
	#this will get the image data.
	def __init__(self, save_path):
		self.images = None
		self.labels = None
		self.groups_list = None
		self.cur_group_index = 0
		self.max_len = None
		self.last_group_list = None
		self.data_savepath = save_path

	def get_group_size(self):
		save_path = self.data_savepath
		with h5py.File(save_path, "r") as file:
			group_size = file["images"]["0"].shape[0]
		return group_size

	def load(self, group_indicies=None):
		"""
		This will load the groups, given the indices
		:param group_indicies: the indicies of the group to load
		:return: 0: success, -1: no path found
		"""
		if not os.path.exists(self.data_savepath):
			print("Must call possible_load_group_indicies first!")
			return -1
		#load the data
		total_images = None
		total_labels = None
		if group_indicies is None: 
			group_indicies = self.groups_list

		with h5py.File(self.data_savepath, "r") as file:
			for v in group_indicies:
				if total_images is None:
					total_images = file["images"][v][()]
					total_labels = file["labels"][v][()]
				else:
					total_images = np.concatenate((total_images, file["images"][v][()]),axis=0)
					total_labels = np.concatenate((total_labels, file["labels"][v][()]),axis=0)
		a = total_images[0]
		self.images = total_images
		self.labels = total_labels
		return 0

	def get_next_group(self, random_selection=True, group_num=1, remove_past_groups=False):
		"""
		This function is an iterator, will iterate through the groups in an hdf5 file.
		:param random_selection: whether to select the group randomly or not.
		:param group_num: The number of groups to load per batch.
		:param remove_past_groups: This is a boolean, if true, will remove the next group number(s) from the iterating
		dataset, otherwise, iterate in a loop, as each get_next_group() is called.
		:return:images, labels.
		"""
		#gets the next group, either a random selection, or increment the list
		assert group_num > 0
		groups=[]
		assert len(self.groups_list), "no more groups, empty groups array."
		for i in range(group_num):
			idx = self.cur_group_index if not random_selection else random.randint(0, len(self.groups_list)-1)
			if not remove_past_groups:
				groups.append(self.groups_list[idx % len(self.groups_list)])
				self.cur_group_index+=1
			else:
				groups.append(self.groups_list.pop(idx % len(self.groups_list)))
		print(groups)
		self.load(groups)
		self.last_group_list = groups
		return self.images, self.labels

	def possible_load_group_indicies(self, shuffle=True, max_len_only=False):
		"""
		This is the possible indices that you can pick a group from.
		:param shuffle: group_indicies the indicies of the group to load
		:param max_len_only: This will force the groups to be of max length.
		:return: groups_list: possible groups to load, -1: no path found
		"""
		#gets the possible groups to load.
		save_path = self.data_savepath
		if not os.path.exists(save_path):
			print("no path found!")
			return -1

		#load the data
		with h5py.File(save_path, "r") as file:
			groups_list= [k for k in file["images"].keys()]
			max_len = max([file["images"][k].attrs["length"] for k in groups_list])
			groups_list = [k for k in file["images"].keys() if not max_len_only or file["images"][k].attrs["length"] >= max_len]
			groups_list = [i for _,i in sorted(zip([int(i) for i in groups_list], groups_list))]
			if shuffle:
				random.shuffle(groups_list)
		self.groups_list = groups_list
		return 0


	def save_file(self, groups=64, dataset_offset=0):
		"""
		loads images from a saved path and sets it as self.images
		Current assumptions:
			- the data and labels are of the same size in the 0th axis
			- the corresponding data and labels are a 1 to 1 mapping. 
		Args:
			groups
				- n sized groups to split the data into.
				- default is 1000 datapoints/group
			dataset_offset
				- This is number to be added to i when saving dataset numbers, only affects the name.
		Returns:
			 0: success
			-1: no data found
		"""
		save_path = self.data_savepath
		if self.images is None and self.labels is None:
			print("no data to save!")
			return -1
		elif self.images is None or self.labels is None:
			item_unavailable = "images" if self.images is None else self.labels
			print("Warning! %s not loaded!"%item_unavailable)
		else:
			pass
		assert len(self.labels) == len(self.images)

		with h5py.File(save_path, "a") as file:
			if not "labels" in file:
				labels_grp = file.create_group("labels")
			else: 
				labels_grp = file["labels"]
			if not "images" in file:
				images_grp = file.create_group("images")
			else: 
				images_grp = file["images"]
			num_data = self.labels.shape[0]
			for i in range(num_data//groups+int(bool(num_data%groups))):
				data_start_index = i*groups
				data_end_index = data_start_index+min(groups, num_data-data_start_index)
				#print("%d"%(i+dataset_offset))
				ldset = labels_grp.create_dataset("%d"%(i+dataset_offset), data=self.labels[data_start_index:data_end_index])
				idset = images_grp.create_dataset("%d"%(i+dataset_offset), data=self.images[data_start_index:data_end_index])
				ldset.attrs["length"] = data_end_index - data_start_index
				idset.attrs["length"] = data_end_index - data_start_index

	def save_by_group(self, labels, filenames, groups):
		"""
		loads in data by groups and saves them accordingly.
		"""
		save_path = self.data_savepath
		num_data = len(filenames)
		num_groups = num_data//groups+int(bool(num_data%groups))
		for i in range(num_groups):
			print("\r"+loading_bar(i, num_groups), end="")
			start_num = i*groups
			self.set_labels(labels[start_num:(i+1)*groups])
			self.get_images_from_filenames(filenames[start_num:(i+1)*groups], False)
			self.save_file(groups, dataset_offset=start_num)

	def set_labels(self, labels):
		#sets the labels
		self.labels = np.asarray(labels)


	def get_images_from_filenames(self, filenames_list, print_loading_bar=True):
		#given a list of filenames, will retrieve the image data
		"""
		Current assumptions:
			- images are of the same size
		
		Args:
			filenames_list:
				- list of the filenames for each of the images.
		"""
		#tf.enable_eager_execution()
		images = None
		#filenames_list = np.asarray(filenames_list).reshape(-1,)
		for i in range(len(filenames_list)):
			if print_loading_bar:
				print("\r"+loading_bar(i, len(filenames_list)), end="")
			image = lycon.load(filenames_list[i])
			#print("MIN, MAX", np.amin(image.numpy()), np.amax(image.numpy()))
			if images is None:
				images = np.zeros((len(filenames_list), *image.shape), np.uint8)
				images[i] = image
			else:
				images[i] = image
		if print_loading_bar:
			print()
		self.images = images
		#tf.disable_eager_execution()


def shuffle_arrays(*args, **kwargs):
	"""
	Takes in arrays of the same length in the 0th axis and shuffles them the same way

	Args:
		*args: numpy arrays.
		**kwargs: numpy arrays.

	Returns:
		arrays in the same order as been put in.
	"""
	args = list(args) + list(kwargs.values())
	idx = np.arange(args[0].shape[0])
	np.random.shuffle(idx)
	new_data = []
	for i in args:
		new_data.append(i[idx])
	return new_data


def upscale2d(x, factor=2):
	##########
	#This function was taken from https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py and modified
	#This is used to reproduce the same upscaling as in the paper, modified to our purposes
	#please check out their work: PGGANs.
	##########
	assert isinstance(factor, int) and factor >= 1
	if factor == 1: return x
	with tf.variable_scope('Upscale2D'):
		s = x.shape
		x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
		x = tf.tile(x, [1, 1, factor, 1, factor, 1])
		x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
	return x

def cross_entropy(inputs, pred, epsilon=1e-7):
	#we need to flatten our data, so we can reduce it per batch.
	inputs = tf.contrib.layers.flatten(inputs)
	pred = tf.contrib.layers.flatten(pred)

	pred = tf.clip_by_value(pred, epsilon, 1-epsilon)
	return -tf.reduce_sum(
			inputs * tf.log(pred) + 
			(1-inputs) * tf.log(1-pred), 
			axis=1)

def kl_divergence(mean, log_var):
	return 0.5*(tf.exp(log_var)+tf.square(mean)-1-log_var)


def create_image_grid(images, aspect_ratio=[1,1]):
	#will create a 2D array of images to be as close to the specified aspect ratio as possible.
	#assumes that images will be able to cover the specified aspect ration min num images = aspect_ratio[1]*aspect_ratio[2]
	#only plots grayscale images right now (can be scaled to multi channel)
	num_images = len(images)
	
	#find the bounding box:
	bounding_box = np.asarray(aspect_ratio)
	while(1):
		#print(bounding_box)
		#print(np.prod(bounding_box), num_images)
		if np.prod(bounding_box) >= num_images:
			break
		bounding_box+=1

	final_image = np.zeros((bounding_box[0]*images.shape[1], bounding_box[1]*images.shape[2], images.shape[3]))
	#fill the available bounding box
	for i in range(num_images):
		row_num = i%bounding_box[0]*images.shape[1]
		col_num = i//bounding_box[0]%bounding_box[1]*images.shape[2]
		final_image[row_num:row_num+images.shape[1], col_num:col_num+images.shape[2]] = images[i,:,:,:]
	return np.squeeze(final_image)

def find_largest_factors(num):
	assert num == num//1
	possible_factors = list(range(int(num//num**0.5+1)))
	possible_factors.reverse()
	for i in possible_factors:
		if num/i == num//i:
			return int(i), int(num/i)
