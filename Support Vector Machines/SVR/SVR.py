from createarrays import *
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

trainingdir = "Gen 2/MRI Data/Testing/Training/"
mri_X_training, mri_y_training = create_training_data(trainingdir)

testdir= "Gen 2/MRI Data/Testing/Test/"
mri_X_test, mri_y_test = create_training_data(testdir)


# Create linear regression object
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

# Train the model using the training sets
regr.fit(mri_X_training, mri_y_training)

# Make predictions using the testing set
mri_y_pred = regr.predict(mri_X_test)

# The coefficients
#print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(mri_y_test, mri_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(mri_y_test, mri_y_pred))