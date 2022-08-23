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
There are multiple models in this project. I will be describing the one that performed the best.  
There are 4 modules. Each module contains 3 layers, a 3D convolutional layer, a 3D max pooling layer, and a batch normalization layer. The modules had convolutional layers with 8, 8, 16, and 32 filters respectively. All of the max pooling layers were 2x2x2. This was all fed into a global average pooling layer and then to a fully connected layer with 64 neurons. This was then connected to a dropout layer of 0.3 and then a fully connected layer with 1 neuron and an activation of sigmoid.

Optimizers and Loss Function
---------------------
Loss function was binary cross-entropy. Optimizer was Adam. A batch size of 2, 50 epochs, a learning rate of 2e-4. A learning rate reducer to 1e-5 over the 50 epochs. Early stopping of training after 30 epochs of no change. Each batch was shuffled before each training.

Conclusion and 
----------------------
The Convolutional Neural Network performed the best. It had the largest AUC and was most accurate in diagnosing patients with and without Multiple Sclerosis. In further research more complicated models could be used. Also Recurrent Neural Networks or CycleGANs could also have potential. Moreover, adding in other parameters including sex, age, ethnicity, bloodtype, genetics could greatly increase the complexity of such a model but could potentially improve its accuracy.

Bibliography
---------------------
Ben Dickson. “What Are Convolutional Neural Networks (CNN)?” TechTalks, TechTalks, 6 Jan. 2020, bdtechtalks.com/2020/01/06/convolutional-neural-networks-cnn-convnets/. 
BrainWeb: Simulated Brain Database, http://www.bic.mni.mcgill.ca/brainweb/. 
Brownlee, Jason. “How Do Convolutional Layers Work in Deep Learning Neural Networks?” Machine Learning Mastery, 16 Apr. 2020, machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/. 
Contributor, TechTarget. “What Is a Convolutional Neural Network? - Definition from WhatIs.com.” SearchEnterpriseAI, TechTarget, 26 Apr. 2018, searchenterpriseai.techtarget.com/definition/convolutional-neural-network. 
DK;, Traboulsee AL;Li. “The Role of MRI in the Diagnosis of Multiple Sclerosis.” Advances in Neurology, U.S. National Library of Medicine, pubmed.ncbi.nlm.nih.gov/16400831/. 
freeCodeCamp.org. “An Intuitive Guide to Convolutional Neural Networks.” FreeCodeCamp.org, FreeCodeCamp.org, 26 Feb. 2018, www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/. 
(Kyle), Korkrid Akepanidtaworn. “Breaking down Classification Evaluation Metrics.” Medium, Medium, 21 Sept. 2019, https://kyleake.medium.com/classification-evaluation-scheme-the-breakdown-of-confusion-matrix-7b8066e978aa. 
“Multiple Sclerosis (MS): Types, Symptoms, and Causes.” Medical News Today, MediLexicon International, www.medicalnewstoday.com/articles/37556. 
“Multiple Sclerosis.” Mayo Clinic, Mayo Foundation for Medical Education and Research, 12 June 2020, www.mayoclinic.org/diseases-conditions/multiple-sclerosis/symptoms-causes/syc-20350269. 
phadji19. “Datasets.” EHealth Lab, Department of Computer Science, University of Cyprus, www.medinfo.cs.ucy.ac.cy/index.php/facilities/32-software/218-datasets. 
Prabhu. “Understanding of Convolutional Neural Network (CNN) - Deep Learning.” Medium, Medium, 21 Nov. 2019, medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148. 
“Random Forest: Introduction to Random Forest Algorithm.” Analytics Vidhya, 24 June 2021, https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/. 
Rolak, Loren A. “Multiple Sclerosis: It's Not the Disease You Thought It Was.” Clinical Medicine & Research, Marshfield Clinic Research Foundation, Jan. 2003, www.ncbi.nlm.nih.gov/pmc/articles/PMC1069023/. 
Szabłowski, Bartosz. “How Convolutional Neural Network Works.” Medium, Towards Data Science, 16 Nov. 2020, towardsdatascience.com/how-convolutional-neural-network-works-cdb58d992363. 
“Understanding Multiple Sclerosis (MS).” Healthline, Healthline, 15 May 2020, www.healthline.com/health/multiple-sclerosis. 
“What Is MS?” National Multiple Sclerosis Society, www.nationalmssociety.org/What-is-MS. 
