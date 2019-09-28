import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

lab = nib.load('Task04_Hippocampus/Task04_Hippocampus/labelsTr/hippocampus_026.nii.gz')

lab_info = lab.get_fdata()

print(lab_info.shape)

print(np.unique(lab_info))
