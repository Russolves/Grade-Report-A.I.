#This program is in script mode and is able to parse through csv files line by line instead of through character recognition
#This program assumes that the header section separates the comment section and the report section

#First import the csv module then the pandas module
import csv
import pandas as pd
import statistics

#importing Machine Learning modules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
#Importing the necessary neural network modules
from sklearn.impute import KNNImputer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

#Import other modules
import numpy as np
import os

#Create the csvfile object using the open() function on "read-mode"
def create(csv):
    #Attempting to read in the .csv file
    print(f"Reading in .csv file: {csv}")
    filename = csv

    #Reading the .csv file
    csvfile = open (filename, "r")

    # #Asking user for which year the .csv file belongs to, can be placed in other program that imports this program
    # userinputyear = input("Please input for which year this student report is from (for e.g. 2019, 2020, 2021...): ").strip()
    # while not userinputyear.isdigit():
    #     print("Invalid year entered, please re-enter the year")
    #     userinputyear = input("Please input for which year this student report is from (for e.g. 2019, 2020, 2021...): ").strip()
    # userinputyear = int(userinputyear)


    #Function that parses through the name of the .csv and then returns the year group
    yeargroup = -1
    for i in filename:
        if i.isdigit():
            yeargroup = i
            break
    #If year group is negative one, ask user to manually enter which year group this is
    if yeargroup == -1:
        yeargroup = int(input("Year Group was not found from filename, please manually key in the year group this file belongs to: "))
    
    #Function that asks the user which semester this grade report belongs to
    semester = input("Please enter for which term this file belongs to (Enter 1, 2 or 3): ")
    #While loop that keeps the user answering unless the answer is 1, 2 or 3
    while True:
        if semester.isdigit():
            semester = int(semester)
            if semester == 1 or semester == 2 or semester == 3:
                break
        print("You have not entered a valid semester. ")
        semester = input("Please enter for which term this file belongs to (Enter 1, 2 or 3): ")

    return yeargroup, semester, csvfile

#Method that parses through the grade reports, line by line using a generator (next)
def parse(csvfile):

    #Creating an empty list for the header for the reports
    data_header = []
    #Creating an empty list for the Comments above the header
    comments = []
    #Creating an empty list for the Reports section below the header
    reports = []
    #Empty list for Residuals that were left behind during parsing
    residuals = []

    #Create Generator to loop over
    dict_obj = (row for row in csv.reader(csvfile))
    
    #Create enumerator object that will create a list of tuples, allowing for the Header section to be found
    enumerator = enumerate(list(dict_obj))
    #This creates a dictionary with the index as its key and the values as its position
    enumerator_dict = dict(enumerator)
    

    #Initialize and compute variable that represents columns in the list within the values of the dictionary
    columns = 0
    columns = len(enumerator_dict[0])
    #print("Number of columns: "+str(columns))

    #Header Section 
    #By looping through the dictionary row by row, find the element within the list of values from the dictionary to find header ("Stkey" is the keyword)
    for i in range(len(enumerator_dict)):
        for j in range(columns):
            #If-statement to find the "Stkey" element in the list
            if "tkey" in enumerator_dict[i][j] or "STKEY" in enumerator_dict[i][j]:
                #print("The row number in which the header was found: "+str(i))
                header_row = i
    
    #Append the data header to the list
    data_header = enumerator_dict.get(header_row, 'Index of header not found')


    #Comments Section
    #Write method that automatically returns the first row in which the keyword "Ab1" is first found
    def firstline():
        for comment_line in range(header_row):
            for comment_element in range(columns):
                if "Ab1" in enumerator_dict[comment_line][comment_element] or "AB1" in enumerator_dict[comment_line][comment_element] or "ab1" in enumerator_dict[comment_line][comment_element]:
                    return comment_line

    #Plug in the "first comment" variable after calling the method
    firstcomment = firstline()
    #Parse through the dictionary line by line, adding sections into the comment section until coming across header
    for k in range(firstcomment, header_row):
        comments.append(enumerator_dict[k])


    #Reports Section
    #Method that returns the first blank line found at the end of the list
    def lastline():
        #Using the same dictionary, parse through the list after the header row to add to the reports section
        line_count = header_row + 1
        for line in range(header_row+1, len(enumerator_dict)):
            #As long as the first value of the column is not equal to 0
            if (len(enumerator_dict[line][0])) == 0:
                return line
            line_count += 1
        #If at this point the program still hasn't found the last line, which means that the last row of the .csv file ended with values instead of empty rows, we use the line_count instead
        return line_count
    #Initialize variable for the first blank row that the for loop encounters at the end of the list
    last_row = lastline()
    #print("This is the row number of the first blank line at the end of the list: "+str(last_row))

    #Parse through the dictionary once more line by line to append the reports section to the list
    for student in range(header_row+1, last_row):
        reports.append(enumerator_dict[student])


    #Print calls to ensure that sections were all divided correctly
    print()
    #print("Comments List: "+str(comments))
    print()
    #print("Headers for the report: " +str(data_header))
    print()
    #print("Reports Section: "+str(reports))
    print()
    #print("Remaining, unsorted rows" +str(residuals))

    #Creating a DataFrame for the Reports Section
    reports_df = pd.DataFrame(reports, columns = data_header)
    comments_df = pd.DataFrame(comments)
    return comments_df, reports_df, columns


#Cleaning up the DataFrame by dropping columns that contain no values
def removeemptycolumns(dataframe, columns):
    lastcolumnlist = []
    for novalue in range(4):
        for emptycolumn in range(columns):
            if not any(dataframe.iloc[novalue, emptycolumn]):
                lastcolumnlist.append(emptycolumn)

    #Detecting non-consecutive columns and finding out the first column that does not have values
    a = 1
    consecutive = []
    lastcolumnset = list(set(lastcolumnlist))
    for a in range(a, len(lastcolumnset)):
        if lastcolumnset[a] - lastcolumnset[a-1] != 1:
            consecutive.append(lastcolumnset[a])
    #Checking to see if the column between the word count lists is empty
    if '' not in dataframe.columns[14:15]:
        if any(consecutive):
            emptycolumn = statistics.mode(consecutive)
            for b in range(emptycolumn, columns):
                if b in dataframe:
                    dataframe.drop(b, inplace = True, axis = 1)
    

    return dataframe

#Method that slices the Report DataFrame into a reportlist for word frequency-array
def reportslice(report, columns):
    #Defining wordcountlist for finding out the two columns with wordcounts
    wordcountlist = []
    for c in range(columns):
        if report.columns[c] == "wordcount" or report.columns[c] == "Wordcount" or report.columns[c] == "WordCount" or report.columns[c] == "word count":
            wordcountlist.append(c)
    if len(wordcountlist) == 2:
        #Checking if the second wordcount column contains all 0 before pinning astype
        if pd.to_numeric(report.iloc[:, wordcountlist[1]], errors = 'coerce').notnull().all() == True:
            #Converting the two wordcount columns into int32 data types within the DataFrame
            report['wordcount']= report['wordcount'].astype('int')
            #Make sure that both the 2 word count columns are greater than 0 (through interpreting the values as integers)
            if int(report.iloc[0, wordcountlist[0]]) >= 0 and int(report.iloc[0, wordcountlist[1]]) >=0:
                reportlistcolumn = int((wordcountlist[0]+wordcountlist[1])/2)
                reportlist = list(report.iloc[:, reportlistcolumn])

        else:
            #Write function to append report column's values into a list
            reportindex = report.columns.get_loc('Report')
            reportlist = list(report.iloc[:, reportindex])

    else:
        print("2 Word Count columns were not found, locating report comments column through slicing (Beware of Error).")
        reportindex = report.columns.get_loc('Report')
        reportlist = list(report.iloc[:, reportindex])
    
    #Slicing list for Stkey
    stkeylist = list(report.iloc[:, 0])
    #Slicing list for First Name
    firstnamelist = list(report.iloc[:, 3])
    #Slicing the list for class/roll_group
    classlist = list(report.iloc[:, 1])
    return reportlist, stkeylist, classlist, firstnamelist
        

def wordfrequencyarray(reportlist, stkeylist):
    #Creating a TfidfVectorizer Object: tfidf
    tfidf = TfidfVectorizer()
    #Apply fit_transform to the report section in order to acquire Word Frequency-Array
    csr_matrix = tfidf.fit_transform(reportlist)
    #Get the words within the WFA
    reportwords = tfidf.get_feature_names_out()
    #Create normalizer object
    normalizer = Normalizer()
    #Create NMF component object
    nmf = NMF ()
    #Creating a pipeline
    pipeline = make_pipeline(nmf, normalizer)
    #Fit & transform csr_matrix into nmf features
    nmf_features = pipeline.fit_transform(csr_matrix)
    #DataFrame with columns containing words and index consisting of names
    features_df = pd.DataFrame(nmf_features, index = stkeylist, columns = reportwords)
    #print(features_df)
    return features_df


    # # Plotting for the number of components to use (applicable on TruncatedSVD & PCA)
    # fig, ax = plt.subplots()
    # xi = np.arange(1, len(reportwords), step = 1)
    # y = np.cumsum(nmf_features.n_features_in_)
    # plt.ylim(0.0, 1.1)
    # plt.plot(xi, y, marker = 'o', linestyle = '--' , color = 'b')
    # plt.xlabel('Number of Components')
    # plt.xticks(np.arange(0, len(reportwords), step = 1)) # Change from 0-based array index to 1-based label
    # plt.ylabel('Number of Features')
    # plt.title('The number of components needed to explain variance')
    # plt.axhline(y = 0.95, color = 'r', linestyle = '-')
    # plt.text(0.5, 0.85, '95% \cutoff threshold', color = 'red', fontsize = 16)
    # ax.grid(axis = 'x')
    # plt.show()

#Construct method that computes the cosine similarity across the entire report column for student entered
def studentinput(features_df, firstnamelist, stkeynamedict, reference_df, student_interest = None):
    #Look for Duplicates
    uniquenames = set()
    duplicates = []
    for name in firstnamelist:
        if name in uniquenames:
            duplicates.append(name)
        else:
            uniquenames.add(name)
    #Computing the number of duplicates within the entire dataset
    counter = {}
    for entry in duplicates:
        if entry not in counter:
            counter[entry] = 0
        counter[entry] += 1
    if student_interest != None:
        userstkey = student_interest
        student = stkeynamedict[userstkey]
    else:
        #Ask for Input
        print("Below are some of the student names from this year group: \n" +str(firstnamelist))
        student = spellcheck()
        #Ensuring that the student is within the DataFrame
        while student not in firstnamelist:
            print("Student was not found within the dataframe, try checking your spelling or make sure that the name of the student you have entered within the year group is correct")
            student = spellcheck()
        print(f"Student {student} was found within the DataFrame!")

        #Should there be duplicates for first name, ask user input to clarify using student keys
        if student in counter:
            for g in counter:
                if g == student:
                    print()
                    print(str(int(counter[g])+1) + " " + student + "s were found in the same year")
                    print(f"Please clarify which {student} you mean through their stkeys shown below: ")
                    #Here, the dictionary comprehension loops over the keys of the dictionary and then finds the key that matches the value the user inputs
                    duplicatekeys = {i for i in stkeynamedict if stkeynamedict[i] == student}
                    duplicatekeyslist = list(duplicatekeys)
                    print(duplicatekeyslist)
                    #Ask user for input on clarification
                    print()
                    print(f"Please enter the student key corresponding to which {student}'s student key you are referring to ")
                    stkeyinput = input("Enter the student key (as shown in the options above, case does not matter): ")
                    stkeyinput = stkeyinput.strip().upper()
                    while stkeyinput not in duplicatekeyslist:
                        print()
                        print(f"You have not entered a valid student key, please re-enter the stkey corresponding to which {student}'s student key you are referring to ")
                        print(duplicatekeyslist)
                        stkeyinput = input("Enter the student key (as shown in the options above, case does not matter): ")
                        stkeyinput = stkeyinput.strip().upper()
                    print()
                    userstkey = stkeyinput
        else:
            stkeyseries = {i for i in stkeynamedict if stkeynamedict[i] == student}
            userstkey = str(list(stkeyseries)[0])


    print(f"{student}'s student key: {userstkey} was chosen")

    #Slicing the Dataframe based on student found
    studentofinterest = features_df.loc[userstkey]        
    #Computing cosine similarities 
    similarities = features_df.dot(studentofinterest)

    #Combining the features DataFrame and the reference DataFrame
    combined_df = pd.merge(similarities.to_frame(), reference_df, left_index = True, right_index = True)
    combined_df = combined_df.sort_values(by = [0], ascending = False)
    return combined_df, userstkey



def spellcheck():
    print()
    student = input("Please enter the first name of student of interest: ")
    student = student.strip().capitalize()
    #Initializing the special name variable
    special = list(student)
    for e in range(len(special)):
        #Processing the input so that spaces and capitalization will not be a mistake
        if special[e] == " ":
            special[e+1] = special[e+1].upper()
            #special = student.replace(special[letter+1], special[letter+1].upper())
        if special[e] == "-":
            special[e+1] = special[e+1].upper()
            #special = student.replace(special[letter+1], special[letter+1].upper())
    student = ''.join(special)
    return student


def referencedf(stkeylist, classlist, firstnamelist):
    reference_df = pd.DataFrame({'First Name': firstnamelist, 'Class': classlist}, index = stkeylist)
    #Dictionary containing the First Names of students as keys and Stkey as values
    stkeynamedict = {stkeylist[i]: firstnamelist[i] for i in range(len(firstnamelist))}
    return reference_df, stkeynamedict

#Method containing studentinput method that asks for the best and worst student name inputs, then creates and returns an average of the entire year
def bestandworst(features_df, firstnamelist, stkeynamedict, reference_df, beststudents, worststudents):
    #Initialize an empty dataframe for the year group
    cohort = pd.DataFrame()
    if any(beststudents or worststudents):
        if any(beststudents):
            #Figuring out if the user wants to use the previous best students list
            best_in = 1
            for i in range(len(beststudents)):
                #If conditional statement checking if one student within the list is not in the firstnamelist, if not then best_in = -1 and break
                if not beststudents[i] in stkeynamedict.keys():
                    best_in = -1
                    break

            if best_in == 1:
                print("Students from previous best students entered found within the file:")
                for j in range(len(beststudents)):
                    print(stkeynamedict[beststudents[j]])
                user_in = input("Would you like to use the previous file's best students as input (Enter Y or N)? ").upper()
                while user_in != "Y" and user_in != "N":
                    print("You have not entered the correct option, please enter the option again")
                    user_in = input("Enter Y or N: ").upper()
                #If user desires for the previous list to be inputted
                if user_in == "Y":
                    bestsum = []
                    best_num = len(beststudents)
                    for x in range(len(beststudents)):
                        print()
                        combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df, beststudents[x])
                        #Combining the Dataframes into one dataframe based on stkey
                        cohort = pd.concat([cohort, combined_df[0]], axis = 1)
                        print()
                        bestsum.append(combined_df[0])
                    cohort["bestsum"] = sum(bestsum)
                #If user wants to enter a new list
                else:
                    #Ask for how many best and worst students the user would like to enter
                    best_num = input("Please enter the number of best students you would like to enter for this year group: ").strip()
                    while True:
                        if best_num.isdigit():
                            best_num = int(best_num)
                            if best_num > 0:
                                break
                        print("You have not entered a valid number")
                        best_num = input("Please enter the number of best students you would like to enter for this year group: ").strip()

                    #Initializing an empty list for bestsum
                    bestsum = []
                    for i in range(best_num):
                        print(f"Best student {i+1} of {best_num}")
                        combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df)
                        #Appending the best students into a list for future use
                        beststudents.append(student)
                        #Combining the Dataframes into one dataframe based on stkey
                        cohort = pd.concat([cohort, combined_df[0]], axis = 1)
                        print()
                        print()
                        bestsum.append(combined_df[0])
                    cohort["bestsum"] = sum(bestsum)
            else:
                #Ask for how many best and worst students the user would like to enter
                best_num = input("Please enter the number of best students you would like to enter for this year group: ").strip()
                while True:
                    if best_num.isdigit():
                        best_num = int(best_num)
                        if best_num > 0:
                            break
                    print("You have not entered a valid number")
                    best_num = input("Please enter the number of best students you would like to enter for this year group: ").strip()

                #Initializing an empty list for bestsum
                bestsum = []
                for i in range(best_num):
                    print(f"Best student {i+1} of {best_num}")
                    combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df)
                    #Appending the best students into a list for future use
                    beststudents.append(student)
                    #Combining the Dataframes into one dataframe based on stkey
                    cohort = pd.concat([cohort, combined_df[0]], axis = 1)
                    print()
                    print()
                    bestsum.append(combined_df[0])
                cohort["bestsum"] = sum(bestsum)
    
        else:
            #Ask for how many best and worst students the user would like to enter
            best_num = input("Please enter the number of best students you would like to enter for this year group: ").strip()
            while True:
                if best_num.isdigit():
                    best_num = int(best_num)
                    if best_num > 0:
                        break
                print("You have not entered a valid number")
                best_num = input("Please enter the number of best students you would like to enter for this year group: ").strip()

            #Initializing an empty list for bestsum
            bestsum = []
            for i in range(best_num):
                print(f"Best student {i+1} of {best_num}")
                combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df)
                #Appending the best students into a list for future use
                beststudents.append(student)
                #Combining the Dataframes into one dataframe based on stkey
                cohort = pd.concat([cohort, combined_df[0]], axis = 1)
                print()
                print()
                bestsum.append(combined_df[0])
            cohort["bestsum"] = sum(bestsum)

        #If there is something in the worststudents list entered through a previous run of the function
        if any(worststudents):
            #Figuring out if the user wants to use the previous worst students list
            worst_in = 1
            for i in range(len(worststudents)):
                #If conditional statement checking if one student within the list is not in the firstnamelist, if not then best_in = -1 and break
                if not worststudents[i] in stkeynamedict.keys():
                    worst_in = -1
                    break
            
            if worst_in == 1:
                print("Students from previous worst students entered found within the file:")
                for j in range(len(worststudents)):
                    print(stkeynamedict[worststudents[j]])
                user_in = input("Would you like to use the previous file's worst students as input (Enter Y or N)? ").upper()
                while user_in != "Y" and user_in != "N":
                    print("You have not entered the correct option, please enter the option again")
                    user_in = input("Enter Y or N: ").upper()
                #If user desires for the previous list to be inputted
                if user_in == "Y":
                    worstsum = []
                    worst_num = len(worststudents)
                    for x in range(len(worststudents)):
                        combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df, worststudents[x])
                        #Combining the Dataframes into one dataframe based on stkey
                        cohort = pd.concat([cohort, combined_df[0]], axis = 1)
                        print()
                        worstsum.append(combined_df[0])
                    cohort["worstsum"] = sum(worstsum)
                #If user wants to enter a new list
                else:
                    #Do the same thing for the worst students of the cohort
                    worst_num = input("Please enter the number of worst students you would like to enter for this year group: ").strip()
                    while True:
                        if worst_num.isdigit():
                            worst_num = int(worst_num)
                            if worst_num > 0:
                                break
                        print("You have not entered a valid number")
                        worst_num = input("Please enter the number of worst students you would like to enter for this year group: ").strip()

                    #Initializing an empty list for worstsum
                    worstsum = []
                    for i in range(worst_num):
                        print(f"Worst student {i+1} of {worst_num}")
                        combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df)
                        #Appending the worst students into a list for later
                        worststudents.append(student)
                        #Combining the Dataframes into one dataframe based on stkey
                        cohort = pd.concat([cohort, combined_df[0]], axis = 1)
                        print()
                        print()
                        worstsum.append(combined_df[0])
                    cohort["worstsum"] = sum(worstsum)
            else:
                #Do the same thing for the worst students of the cohort
                worst_num = input("Please enter the number of worst students you would like to enter for this year group: ").strip()
                while True:
                    if worst_num.isdigit():
                        worst_num = int(worst_num)
                        if worst_num > 0:
                            break
                    print("You have not entered a valid number")
                    worst_num = input("Please enter the number of worst students you would like to enter for this year group: ").strip()

                #Initializing an empty list for worstsum
                worstsum = []
                for i in range(worst_num):
                    print(f"Worst student {i+1} of {worst_num}")
                    combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df)
                    #Appending the worst students into a list for later
                    worststudents.append(student)
                    #Combining the Dataframes into one dataframe based on stkey
                    cohort = pd.concat([cohort, combined_df[0]], axis = 1)
                    print()
                    print()
                    worstsum.append(combined_df[0])
                cohort["worstsum"] = sum(worstsum)
        else:
            #Do the same thing for the worst students of the cohort
            worst_num = input("Please enter the number of worst students you would like to enter for this year group: ").strip()
            while True:
                if worst_num.isdigit():
                    worst_num = int(worst_num)
                    if worst_num > 0:
                        break
                print("You have not entered a valid number")
                worst_num = input("Please enter the number of worst students you would like to enter for this year group: ").strip()

            #Initializing an empty list for worstsum
            worstsum = []
            for i in range(worst_num):
                print(f"Worst student {i+1} of {worst_num}")
                combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df)
                #Appending the worst students into a list for later
                worststudents.append(student)
                #Combining the Dataframes into one dataframe based on stkey
                cohort = pd.concat([cohort, combined_df[0]], axis = 1)
                print()
                print()
                worstsum.append(combined_df[0])
            cohort["worstsum"] = sum(worstsum)
    else:
        #Ask for how many best and worst students the user would like to enter
        best_num = input("Please enter the number of best students you would like to enter for this year group: ").strip()
        while True:
            if best_num.isdigit():
                best_num = int(best_num)
                if best_num > 0:
                    break
            print("You have not entered a valid number")
            best_num = input("Please enter the number of best students you would like to enter for this year group: ").strip()

        #Initializing an empty list for bestsum
        bestsum = []
        for i in range(best_num):
            print(f"Best student {i+1} of {best_num}")
            combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df)
            #Appending the best students into a list for future use
            beststudents.append(student)
            #Combining the Dataframes into one dataframe based on stkey
            cohort = pd.concat([cohort, combined_df[0]], axis = 1)
            print()
            print()
            bestsum.append(combined_df[0])
        cohort["bestsum"] = sum(bestsum)

        #Do the same thing for the worst students of the cohort
        worst_num = input("Please enter the number of worst students you would like to enter for this year group: ").strip()
        while True:
            if worst_num.isdigit():
                worst_num = int(worst_num)
                if worst_num > 0:
                    break
            print("You have not entered a valid number")
            worst_num = input("Please enter the number of worst students you would like to enter for this year group: ").strip()

        #Initializing an empty list for worstsum
        worstsum = []
        for i in range(worst_num):
            print(f"Worst student {i+1} of {worst_num}")
            combined_df, student = studentinput(features_df, firstnamelist, stkeynamedict, reference_df)
            #Appending the worst students into a list for later
            worststudents.append(student)
            #Combining the Dataframes into one dataframe based on stkey
            cohort = pd.concat([cohort, combined_df[0]], axis = 1)
            print()
            print()
            worstsum.append(combined_df[0])
        cohort["worstsum"] = sum(worstsum)
    
    #Applying the algorithm to the dataframe, inserting two columns for the best student's sums and worst student's sum and a last column of overall computed scores
    cohort["overall"] = 0.5*((cohort["bestsum"]/best_num)+(1/(worst_num*cohort["worstsum"])))
    overall_range = max(cohort["overall"]) - min(cohort["overall"])
    cohort["final"] = (cohort["overall"] - min(cohort["overall"]))/overall_range

    #Returning the "final" column of the DataFrame and the best & worst students list
    return cohort["final"], beststudents, worststudents

#Sub-Main Script that runs for each .csv file that is inputted into the program, also constructs a large dataframe
def subscript(csv, large_df, cohort_rows, beststudents, worststudents):
    yeargroup, semester, csvfile = create(csv)
    #Parsing the csv file to acquire 3 things
    comments, report, columns = parse(csvfile)
    #Removing Any Extra Columns
    comments = removeemptycolumns(comments, columns)
    report = removeemptycolumns(report, columns)
    #Slicing Report DataFrame
    reportlist, stkeylist, classlist, firstnamelist = reportslice(report, columns)
    #First create word frequency array
    features_df = wordfrequencyarray(reportlist, stkeylist)
    #Creating the second dataframe for reference
    reference_df, stkeynamedict = referencedf(stkeylist, classlist, firstnamelist)
    score_column, beststudents, worststudents = bestandworst(features_df, firstnamelist, stkeynamedict, reference_df, beststudents, worststudents)
    
    #Implementing try, except and else to raise error should the dataframes inputted not be of the same cohort
    try:
        large_df = pd.concat([large_df, score_column], axis = 1)
    except:
        print("The file that was just read in by the program has encountered some errors")
    else:
        #Rename the column from the returned dataframe column "score_column"
        columnname = str(yeargroup) + ","+str(semester)
        large_df.rename(columns = {'final':columnname}, inplace = True)
        print("File successfully combined into DataFrame!")
        print()
        #Tracking the shape of the dataframe so that if the two dataframes that are combined together differ by year group, program tells the user that the year group was not from the cohort
        cohort_rows.append(large_df.shape[0])
        if len(cohort_rows) >= 2:
            if cohort_rows[-1] > (cohort_rows[-2] +min(cohort_rows)*0.5):
                print("It seems that the file that was just read in is not from the same year group, choosing to continue can affect the machine learning model's predictive capabilities")
                print("If you do not wish to continue with this excel file, please enter 0 to completely exit the program")
                print("WARNING! If you exit the program you will have to re-start the entire program again and re-input all the best/worst students")
                print()
                print()
                userstop = input("Do you still wish to proceed? (Enter 0 to stop the execution of the program; enter 1 to proceed): ")
                while True:
                    if userstop.isdigit():
                        userstop = int(userstop)
                        if userstop == 0:
                            exit()
                        elif userstop == 1:
                            # temp_shape = cohort_rows.pop(-1)
                            break
                    print("You have not enterd 0 or 1, please re-enter a valid choice")
                    userstop = input("Do you still wish to proceed? (Enter 0 to stop the execution of the program; enter 1 to proceed): ")
    finally:
        return large_df, cohort_rows, beststudents, worststudents

#Imputation method - K-Nearest Neighbor Imputation
def knn_imputation(array):
    imputer = KNNImputer(n_neighbors = 2, weights = "uniform")
    imputed_array = imputer.fit_transform(array)
    return imputed_array

#Applying the Neural Network - Deep Learning
def learn(array):
    predictors = array[:, :-1]
    print(predictors.shape)
    n_cols = predictors.shape[1]
    target = array[:, -1]
    print(target)
    print(target.shape)

    #Building the Deep Learning Model
    model = Sequential()
    model.add(Dense(10000, activation = 'relu', input_shape = (n_cols, )))
    model.add(Dense(8000, activation = 'relu'))
    model.add(Dense(8000, activation = 'relu'))
    model.add(Dense(5000, activation = 'relu'))
    model.add(Dense(5000, activation = 'relu'))
    model.add(Dense(2000, activation = 'relu'))
    model.add(Dense(2000, activation = 'relu'))
    model.add(Dense(1))
    #Compiling the Model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy', 'mse'])
    model.fit(predictors, target, validation_split = 0.2, epochs = 20)

    #Model Summary
    model.summary()
    #Saving the Model for future predictions
    model.save('model_file.h5')

#Prediction Step
my_model = load_model('model_file.h5')
predictions = my_model.predict()

#Main script "blueprints" for program to execute methods
def main():
    #Initializing an empty list for .csv files found within directory
    listofcsv = []
    path = os.getcwd()
    for k in os.listdir(path):
        if k.endswith('.csv'):
            listofcsv.append(k)
    #Initializing the large dataframe for where the year cohort's "final" scores are all stored
    large_df = pd.DataFrame()
    #Initializing an empty list containing the number of rows of the "cohort" DataFrame
    cohort_rows = []
    #Initializing a best & worst student list for the year cohort if there is none
    beststudents = []
    worststudents = []
    for csv in listofcsv:
        large_df, cohort_rows, beststudents, worststudents = subscript(csv, large_df, cohort_rows, beststudents, worststudents)
    print(large_df)
    #Converting the dataframe into a numpy array
    array = large_df.to_numpy()
    print("All .csv files were read into the program. Now beginning prediction step:")
    #For our test run use KNN Imputation, then test to see what yields better value
    imputed_array = knn_imputation(array)
    #Printing out the numpy array and accounting for its shape
    print(imputed_array.shape)
    learn(imputed_array)

#Do not change anything here
if __name__ == "__main__":
    main()