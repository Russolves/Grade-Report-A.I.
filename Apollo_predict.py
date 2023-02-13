#Importing the necessary modules
from Apollo_learn import subscript, folder_reader, knn_imputation
from tensorflow.keras.models import load_model
from statistics import mode
import pandas as pd
import os

#Importing module to guess gender of students
import gender_guesser.detector as gender

#Method to predict using the prediction dataset
def predict(array, stkeylist, pref_namelist):
    my_model = load_model('model_file.h5')
    predictions = my_model.predict(array)
    print(predictions.shape)
    print()
    #Zipping the stkey of the students together with the prediction similarity scores (overall)
    predict_list = predictions.tolist()
    pred = []
    for j in range(len(predict_list)):
        for i in range(len(predict_list[j])):
            pred.append(predict_list[j][i])

    # #Initialize a general value to 'normalize' the other values by
    # subtract = int(mode(pred))
    # #Subtract the mode from the values because the values are too similar to each other
    # sub_list = []
    # for x in pred:
    #     sub_list.append(x - subtract)

    #Perform Normalization so that the data becomes normalized (if deleting the part above, replace 'sublist' with 'pred')
    pred_min = min(pred)
    # print("pred_min "+str(pred_min))
    pred_max = max(pred)
    # print("pred_max "+str(pred_max))
    pred_normal = []
    for k in pred:
        value = (k - pred_min)/(pred_max - pred_min)
        pred_normal.append(value)
    predict_dict = {stkey: sim_score for stkey, sim_score in zip(stkeylist, pred_normal)}
    # print(predict_dict)
    #Zipping the stkey of the students with the preferred name of the students
    pref_dict = {stkey: pref.strip() for stkey, pref in zip(stkeylist, pref_namelist)}
    # print(pref_dict)
    return pred_normal, predict_dict, pref_dict

#Method that will process the reports in the .csv file and then return the reports section
def process(csv):
    #Initializing the large dataframe for where the year cohort's "final" scores are all stored
    process_df = pd.DataFrame()
    #Initializing an empty list containing the number of rows of the "cohort" DataFrame
    cohort_rows = []
    #Initializing a best & worst student list for the year cohort if there is none
    beststudents = []
    worststudents = []
    
    process_df, cohort_rows, beststudents, worststudents, reportlist, stkeylist, classlist, firstnamelist, stkeynamedict, pref_namelist, surnamelist = subscript(csv, process_df, cohort_rows, beststudents, worststudents)
    #Converting the computed 'process_df' containing the overall score into numpy array
    process_array = process_df.to_numpy()
    #Zipping the stkey of the students together with the prediction similarity scores
    process_list = process_array.tolist()
    proc = []
    for j in range(len(process_list)):
        for i in range(len(process_list[j])):
            proc.append(process_list[j][i])
    comp_dict = {stkey: score for stkey, score in zip(stkeylist, proc)}
    #Zipping the stkey of the students together with the reports section
    report_dict = {stkey: report for stkey, report in zip(stkeylist, reportlist)}
    #Zipping the stkey of the students with the preferred name of the students
    pref_dict = {stkey: pref.strip() for stkey, pref in zip(stkeylist, pref_namelist)}
    # print(comp_dict)
    # print(report_dict)
    # print(pref_dict)
    return comp_dict, report_dict, pref_dict

#Method that is able to compare scores between the two groups and return a dictionary containing the stkey of the predicted student as key and the stkey of the comparison student as value
def compare_score(compscore_dict, predict_dict):
    #Initialize the dictionary we are returning (with the key being the student predicted and the value being the student compared)
    match_dict = {}
    for i in predict_dict:
        #Initialize a temporary dictionary for calculating the closest score (through using 'min()' on the absolute value of difference)
        diff = {}
        for j in compscore_dict:
            diff[j] = abs(predict_dict[i] - compscore_dict[j])
        #Obtain the key (stkey) to the value of the closest score
        min_value = min(diff.values())
        closest_student = [k for k in diff if diff[k] == min_value]
        #For each stkey of prediction student, add stkey of the comparison student (with closest score); note that [0] is used because list comprehension returns a list containing the one stkey comparison student value so it has to be sliced
        match_dict[i] = closest_student[0]
    return match_dict

#Method that is able to identify genders given a dictionary of stkeys and preferred names; returns stkey with 'male', 'female' or 'andy' (androgynous)
def gender_identifier(pref_dict):
    d = gender.Detector()
    #Initialize an empty dictionary for dictionary return 
    pref_id = {}
    for key in pref_dict:
        name = pref_dict[key]
        result = d.get_gender(name)
        if result == "mostly_female" or result == "female":
            pref_id[key] = "F"
        elif result == "mostly_male" or result == "male" or result == "andy" or result == "unknown":
            pref_id[key] = "M"
    #print(pref_id)
    return pref_id

#Report comment Method that replaces the name & the pronouns of the students with the ones of interest
def report_replace(match_dict, report_dict, pref_interest, pref_replace, pref_genderinterest, pref_genderreplace):
    #Intialize an empty report dict for final report comment output
    report_output = {}
    #Initialize empty word count dict
    wordcount_dict = {}
    for key in match_dict:
        #Key is stkey of student of interest
        value = match_dict[key]
        #Value is stkey of student we are replacing
        string = report_dict[value]
        #First split the string report into a list (split by spaces)
        string_list = string.split(" ")
        #For loop to iterate over each word in the list string entry
        for x in range(len(string_list)):
            #Replace the student's name with the student of interest's name
            if string_list[x] == pref_replace[value]:
                string_list[x] = pref_interest[key]
            elif string_list[x] == pref_replace[value] + ".":
                string_list[x] = pref_interest[key] + "."
            elif string_list[x] == pref_replace[value] + "'s":
                string_list[x] = pref_interest[key] + "'s"
            elif string_list[x] == pref_replace[value] + "'":
                string_list[x] = pref_interest[key] + "'"
            #Replace the student's pronouns with the student of interest's pronouns
            if pref_genderreplace[value] != pref_genderinterest[key]:
                if string_list[x] == "he":
                    string_list[x] = "she"
                elif string_list[x] == "He":
                    string_list[x] = "She"
                elif string_list[x] == "she":
                    string_list[x] = "he"
                elif string_list[x] == "She":
                    string_list[x] = "He"
                elif string_list[x] == "him":
                    string_list[x] = "her"
                elif string_list[x] == "him.":
                    string_list[x] = "her."
                elif string_list[x] == "her":
                    string_list[x] = "his"
                elif string_list[x] == "Her":
                    string_list[x] = "Him"
                elif string_list[x] == "her.":
                    string_list[x] = "his."
                elif string_list[x] == "his":
                    string_list[x] = "her"
                elif string_list[x] == "His":
                    string_list[x] = "Her"
                elif string_list[x] == "his.":
                    string_list[x] = "hers."
                elif string_list[x] == "hers":
                    string_list[x] = "his"
                elif string_list[x] == "hers.":
                    string_list[x] = "his."
        #Initialize a Word Count Variable
        word_count = len(string_list)
        wordcount_dict[key] = word_count
        #Piece string back together
        string_processed = ' '.join(string_list)
        #Put it into dictionary
        report_output[key] = string_processed
    #print(report_output)
    return report_output, wordcount_dict

#Method that converts dictionaries (with stkeys as keys) into pandas dataframe with stkeys as index
def df_creator(dictionary, name_col):
    #Initialize empty dictionary output, stkey list and value list
    dictionary_df = {}
    stkey_list = []
    value_list = []
    #For loop to iterate over the dictionary inputted
    for stkey in dictionary:
        stkey_list.append(stkey)
        value_list.append(dictionary[stkey])
    #Put it to the dictionary (that will become the dataframe)
    dictionary_df["Stkey1"] = stkey_list
    dictionary_df[name_col] = value_list
    df = pd.DataFrame(dictionary_df)
    df.set_index("Stkey1")
    return df
                    

#Main program
def main():
    print()
    #Initializing a file list 
    filelist = []
    #Retrieving folders within the directory
    original_path = os.getcwd()
    for k in os.listdir(original_path):
        if k.endswith(".csv") or "Prediction" in k or "prediction" in k:
            filelist.append(k)
    print("Welcome to the Prediction program")
    print()
    print("Note that there should be only 1 .csv file and 1 folder named 'Prediction/prediction' in the list shown below")
    print("The folder and .csv file this program will use: "+str(filelist))
    prediction_folder = None
    for j in filelist:
        if "Prediction" in j or "prediction" in j:
            prediction_folder = j
            prediction_path = original_path+"\\" + j
    if prediction_folder == None:
        raise Exception("Check if the folder's name is 'Prediction/prediction' or if the .csv file is not correct, if so please replace the folders and restart the program")
    print(prediction_path)
    #Changing directories to access the prediction file
    os.chdir(prediction_path)
    print(f"Reading in the {prediction_folder} found in directory, containing: "+str(os.listdir(prediction_path)))
    print()
    #Returning the list of csv predictions to make
    listofcsv = folder_reader(prediction_path)
    #Initializing the large dataframe for where the year cohort's "final" scores are all stored
    large_df = pd.DataFrame()
    #Initializing an empty list containing the number of rows of the "cohort" DataFrame
    cohort_rows = []
    #Initializing a best & worst student list for the year cohort if there is none
    beststudents = []
    worststudents = []
    for csv in listofcsv:
        large_df, cohort_rows, beststudents, worststudents, reportlist, stkeylist, classlist, firstnamelist, stkeynamedict, pref_namelist, surnamelist = subscript(csv, large_df, cohort_rows, beststudents, worststudents)
    print(large_df)
    #Converting the dataframe into a numpy array
    array = large_df.to_numpy()
    print("Prediction Set inputted. Initating Model Predictions:")
    #For our test run use KNN Imputation, then test to see what yields better value
    imputed_array = knn_imputation(array)
    #Printing out the numpy array and accounting for its shape
    print(imputed_array.shape)
    #Changing the directory back into the original path so that the model saves in the right place
    os.chdir(original_path)
    predictions, predict_dict, pref_interest = predict(imputed_array, stkeylist, pref_namelist)
    #Look for the .csv file within the original directory where the program is located
    for entry in filelist:
        if entry.endswith(".csv"):
            csv = entry
    compscore_dict, report_dict, pref_replace = process(csv)
    #Call in method that can compare the scores
    match_dict = compare_score(compscore_dict, predict_dict)
    #Identify genders from the names and then put them into a dictionary along with stkeys as the keys
    pref_genderinterest = gender_identifier(pref_interest)
    pref_genderreplace = gender_identifier(pref_replace)
    #Method that is able to replace pronouns and name in the reportlist
    report_dict, wordcount_dict = report_replace(match_dict, report_dict, pref_interest, pref_replace, pref_genderinterest, pref_genderreplace)
    pref_df = df_creator(pref_interest, "PREF_NAME")
    #Initialize final output df
    output_df = pd.DataFrame()
    output_df["Stkey"] = stkeylist
    output_df.set_index("Stkey")
    output_df["Class"] = classlist
    output_df["SURNAME"] = surnamelist
    output_df["FIRST_NAME"] = firstnamelist
    output_df = output_df.join(pref_df, how = 'inner', lsuffix = 'Stkey', rsuffix = 'Stkey1')
    #By setting the index to the next dataframe then resetting it & dropping it
    output_df = output_df.set_index("Stkey1")
    output_df = output_df.reset_index(drop = True)
    #Combine the report_df with the rest of the dataframe
    report_df = df_creator(report_dict, "Report")
    output_df = output_df.join(report_df, how = 'inner', lsuffix = 'Stkey', rsuffix = 'Stkey1')
    output_df = output_df.set_index("Stkey1")
    output_df = output_df.reset_index(drop = True)
    #Add the word count column
    wordcount_df = df_creator(wordcount_dict, "wordcount")
    output_df = output_df.join(wordcount_df, how = 'inner', lsuffix = 'Stkey', rsuffix = 'Stkey1')
    output_df = output_df.set_index("Stkey1")
    output_df = output_df.reset_index(drop = True)

    print(output_df)

    #Converting the dataframe to .csv
    output_df.to_csv("prediction_output.csv", index = False)

#Do not change anything here
if __name__ == "__main__":
    main()
