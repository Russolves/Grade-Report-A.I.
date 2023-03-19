# Grade-Report-A.I.
Apollo_learn program that can take in large amounts of .csv files in the same directory and trains a deep learning neural network model based on the best/worst students evaluated by the user. Apollo_predict program then utilizes the model from the  first program to produce a projected report for each individual student inputted.

Note that the .csv files read in have to be in a specific format as specifed below:
stkey (student key unique to each student)---Class---Surname---First Name---Pref Name (Preferred name)---Gender (M or F)---Report (Containing comments)---Word Count

It is recommended that the user first first separate the students into cohorts as folders where each folder contains the reports of the same group of students over a period of different years/semesters in the form of .csv files. Doing this will ensure that the deep learning model has a consistent group of students as basis to make a comparison. Training a model based on a consistent dataset will yield a much higher accuracy in predicting the student's future performances as well.
Important note: The number of grade reports (.csv files) present within the different cohort folders must be consistent as well or the reader method will not have a consistent reference point on which the overall training DataFrame is constructed.


Prediction Component
The program also requires a specific group of students as inputs to make a prediction. This means reading in a .csv file in the same format as the .csv files within the different cohort folders present within the same directory. Apollo_predict would then use the students names from the first report to make a prediction, outputting the following:

---Stkey---Class---Surname---First Name---Pref Name---Report (Predicted Comments)---wordcount---

*User may need to manually change student's class for which the new report is written in the .csv file output

The report produced would be named 'prediction_output', user then would have to remove the file from within the directory to prevent the .csv file from being overwritten by next predictions.


Like all machine learning models, dataset would require a huge dataset upon which to operate or otherwise prediction accuracy would not be high. Future optimization updates could include the incorporated functionality of adding in transfer students from within the program and the ability to customize features of the predicted student report output directly through the program (like class). In addition functionalities that can automatically sort .csv files within the directory into folders and then make predictions is also a considered direction for the project. 
