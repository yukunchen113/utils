#this contains all the utils needed for machine learning projects
import tensorflow as tf 
import numpy as np
import os
import gzip
import pickle
import scipy as scp
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
def get_celeba_data(datapath, save_new=False):
	#loads from numpy file, if available or specified, else save to numpy file
	datapath = os.path.join(datapath, "celeba")
	labels_file = "list_attr_celeba.txt"
	images_saved_file = "images_saved.hdf5"
	images_saved_path = os.path.join(datapath, images_saved_file)
	labels_path = os.path.join(datapath, labels_file)
	images_path = os.path.join(datapath, "images")

	dataset = get_data()
	if not save_new:
		ret = dataset.load(images_saved_path)
		if ret:
			save_new = True

	if save_new:
		#get the labels and images filenames:
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
		
		dataset.set_labels(labels)
		dataset.get_images_from_filenames(filenames)
		dataset.save_file(images_saved_path)

	return dataset.images, dataset.labels

	


class get_data():
	#this will get the image data.
	def __init__(self):
		self.images = None
		self.labels = None

	def load(self, save_path):
		"""
		loads images from a saved path and sets it as self.images
		Args:
			save_path
				- the path of the saved file
		Returns:
			 0: success
			-1: no path found
		"""
		if not os.path.exists(save_path):
			print("no path found!")
			return -1
		#load the data
		with h5py.File(save_path, "r") as file:
			self.images = file["images"][()]
			self.labels = file["labels"][()]
		return 0

	def save_file(self, save_path, groups=1000):
		"""
		loads images from a saved path and sets it as self.images
		Current assumptions:
			- the data and labels are of the same size in the 0th axis
			- the corresponding data and labels are a 1 to 1 mapping. 
		Args:
			save_path
				- the path of the saved file
			groups
				- n sized groups to split the data into.
				- default is 1000 datapoints/group
		Returns:
			 0: success
			-1: no data found
		"""
		if self.images is None and self.labels is None:
			print("no data to save!")
			return -1
		elif self.images is None or self.labels is None:
			item_unavailable = "images" if self.images is None else self.labels
			print("Warning! %s not loaded!"%item_unavailable)
		else:
			pass
		assert len(self.labels) == len(self.images)
		with h5py.File(save_path, "w") as file:
			idset = file.create_dataset("images", data=self.images)
			ldset = file.create_dataset("labels", data=self.labels)




	def set_labels(self, labels):
		#sets the labels
		self.labels = np.asarray(labels)


	def get_images_from_filenames(self, filenames_list):
		#given a list of filenames, will retrieve the image data
		"""
		Current assumptions:
			- images are of the same size
		
		Args:
			filenames_list:
				- list of the filenames for each of the images.
		"""
		images = None
		for i in range(len(filenames_list)):
			print(loading_bar(i, len(filenames_list)), end="\r")
			image = lycon.load(filenames_list[i])
			if images is None:
				images = np.zeros((len(filenames_list), *image.shape), np.int8)
				images[i] = image
			else:
				images[i] = image
		print()
		self.images = images
		

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
	return 0.5*tf.reduce_sum(tf.exp(log_var)+tf.square(mean)-1-log_var,axis=1)

def create_image_grid(images, aspect_ratio=[1,1]):
	#will create a 2D array of images to be as close to the specified aspect ratio as possible.
	#assumes that images will be able to cover the specified aspect ration min num images = aspect_ratio[1]*aspect_ratio[2]
	#only plots grayscale images right now (can be scaled to multi channel)
	num_images = len(images)
	
	#find the bounding box:
	bounding_box = np.asarray(aspect_ratio)
	while(1):
		if np.prod(bounding_box) >= num_images:
			break
		bounding_box+=1

	final_image = np.zeros((bounding_box[0]*images.shape[1], bounding_box[1]*images.shape[2]))
	#fill the available bounding box
	for i in range(num_images):
		row_num = i%bounding_box[0]*images.shape[1]
		col_num = i//bounding_box[0]%bounding_box[1]*images.shape[2]
		final_image[row_num:row_num+images.shape[1], col_num:col_num+images.shape[2]] = images[i,:,:,0]
	return final_image