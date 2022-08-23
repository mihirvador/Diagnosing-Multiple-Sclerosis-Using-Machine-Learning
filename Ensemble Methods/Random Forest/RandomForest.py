from createarrays import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score, accuracy_score

trainingdir = "Gen 2/MRI Data/Monke/Training"
mri_X_training, mri_y_training = create_training_data(trainingdir)

testdir= "Gen 2/MRI Data/Monke/Test"
mri_X_test, mri_y_test = create_training_data(testdir)


# Create linear regression object
regr = RandomForestClassifier(random_state=0, n_jobs=-1)

# Train the model using the training sets
regr.fit(mri_X_training, mri_y_training)

# Make predictions using the testing set
mri_y_pred = regr.predict(mri_X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(mri_y_test, mri_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(mri_y_test, mri_y_pred))
# ROC-AUC
print("ROC-AUC: %.2f" % roc_auc_score(mri_y_test, mri_y_pred))
# Accuracy
print("Accuracy: %.2f" % accuracy_score(mri_y_test, mri_y_pred))
# Precision
print("Precision: %.2f" % precision_score(mri_y_test, mri_y_pred))
# Recall
print("Recall: %.2f" % recall_score(mri_y_test, mri_y_pred))