# Diagnosing-Multiple-Sclerosis-Using-Machine-Learning

The purpose of this repository is to showcase various Machine Learning algorithms to diagnose Multiple Sclerosis from analyzing 3D MRI scans.


**Algorithms/Models**

Ordinary Least Squares

Ridge  
Lasso  
Elastic-Net  
LassoLars  
Kernel Ridge  
SVR  
NuSVR  
LinearSVR  
DecisionTreeClassifier  
Extra Trees  
Gradient Boosting Estimator  
Hist Gradient Boosting Classifier  

Best Models  
Convolutional Neural Network  
Random Forests  


**Performace**

| Model | Ordinary Least Squares | Ridge | Lasso | Elastic-Net | LassoLars | Kernel Ridge | SVR | NuSVR | LinearSVR | DecisionTreeClassifier | Extra Trees | Gradient Boosting Estimator | Hist Gradient Boosting Classifier |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Mean Squared Error | 0.24 | 0.24 | 0.03 | 0.03 | 0.25 | 0.24 | 0.25 | 0.25 | 0.25 | 0.33 | 0.42 | 0.33 | 0.33 |
| Coefficient of Determination | 0.04 | 0.04 | 0.86 | 0.90 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | -0.33 | -0.67 | -0.33 | -0.33 |

| Best Models | Convolutional Neural Network | Random Forests |
| ------- | ------- | ------- |
| Test Area Under the ROC Curve | 0.8643 | 0.5580 |
| Test Accuracy | 0.7700 | 0.5710 |
| Test Precision | 0.6957 | 0.5750 |
| Test Recall | 0.9600 | 0.6030 |

Closer mean squared error is to 0 the better the algorithm performed.  
Closer coefficient of determination is to 1 the better the algorithm performed.  
Closer test area under the ROC curve is to 1 the better the algorithm performed.  
Closer test accuracy is to 1 the better the algorithm performed.  
Closer test precision is to 1 the better the algorithm performed.  
Closer test recall is to 1 the better the algorithm performed.  

Usage
-------------------------------------------------------
There are 2 files in every folder, one is createarrays.py, the other is "algorithmname".py. In "algorithmname".py modify trainingdir and testdir to point to the training and test dataset folders respecively, then run it to train and test the model.  
The only exception is the Convolutional Neural Network Folder. There are multiple files in that folder, but run Custom.py. The first time, modify noMS_traindir, MS_traindir, noMS_testdir, and MS_testdir to point to the training and test dataset folder for MRI scans with and without multiple sclerosis, Then type "c" to create a Tensorflow record. Run it again, and type "t" to train the model. Run it a third time, and type "p" to test the model. The model can be changed by selected the model by modifying the modelnum to a number 1-6. The models are stored in Models.py. 

Version
-------------------------
This was all written in TensorFlow Keras 2 in Python 3.7.0 or in SciKit in Python 3.7.0

Dataset
-------------------
3000 3D MRI Scans from McGill University’s Anonymous BrainWeb. 1500 scans with no Multiple Sclerosis, 500 scans with mild Multiple Sclerosis lesions, 500 with moderate Multiple Sclerosis lesions, and 500 with severe Multiple Sclerosis lesions.

Preprocessing and Creating Training and Test Sets
--------------------------------
Reshape all MRIs to 181x217x181 and normalize the volues to between 0 and 1.  
**Convolutional Neural Network:**
Convert MRIs to tensors, shuffle and split set into 56% for training, 24% for validation, and 20% of testing.  
**All other methods:**
Convert MRIs to NumPy arrays. Flatten these arrays. Shuffle them and split set into 80% for training and 20% for test.

Creating Neural Network
-------------------------------------------
