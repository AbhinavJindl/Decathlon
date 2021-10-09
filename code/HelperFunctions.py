


import pickle
import json
import os

def getModalityType(taskDir):
	ifile = open(os.path.join('../Dataset',taskDir, 'dataset.json'),'r')
	data = ifile.read()
	jsonData = json.loads(data)
	return list(jsonData['modality'].items())[0][1]
'''
	img : 4d image where 4th dimension is the modality
	axis : axis along slice has to be extracted
	sliceNumber : The slice Number (integer)
	modalityNumber : modality of the slice (integer) (-1 indicates it is 3d image)
'''
def getSliceFromImage(img, axis, sliceNumber, modalityNumber=-1):
	if(axis == 0):
		if modalityNumber == -1:
			return img[sliceNumber,:,:]
		return img[sliceNumber,:,:,modalityNumber]
	elif(axis == 1):
		if modalityNumber == -1:
			return img[:,sliceNumber,:]
		return img[:,sliceNumber,:,modalityNumber]
	elif(axis == 2):
		if modalityNumber == -1:
			return img[:,:,sliceNumber]
		return img[:,:,sliceNumber,modalityNumber]
	else:
		return None


def save_pickle(obj,filename):
	with open(filename, 'wb') as handle:
		pickle.dump(obj, handle)

def load_pickle(filename):
	with open(filename, 'rb') as handle:
		return pickle.load(handle)

