import os
import numpy as np
import nibabel as nib
import sklearn

def create_training_data(datadir):
    X = []
    y = []
    cats = ["noMS", "MS"]
    for category in cats:
        path = os.path.join(datadir,category)
        class_num = cats.index(category)  # get the classification  (0 or a 1). 0=no 1=yes
        for mri in os.listdir(path):  # iterate over each mri
            try:
                temp_array = nib.load(os.path.join(path,mri))  # convert to array
                mri_array = temp_array.get_fdata()
                mri_array = mri_array/1500
                mri_array = np.array(mri_array)
                mri_array = mri_array.flatten()
                X.append(mri_array)
                y.append(class_num)  # add this to our data array
            except Exception as e:
                pass
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int8")
    X, y = sklearn.utils.shuffle(X, y)
    return X, y