#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini

14 March 2022

KAI ZEN
"""

attr_list = ["IM_S_Ent", "IM_T_Ent", "IM_R_Ent", "IM_F_Inf"]
#%% GENERAL SETTINGS

# Packages to import
import pickle
import os
import numpy as np
import itertools
# GRID SEARCH
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from class_signal import Signal 

# Import the necessary modules
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier



def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def data_classification(obj_path_list, n_conc, patient, flag_channels):

    ### INITIALIZATION
    dataset_samples = []
    dataset_labels = []
    obj_names = []

    ### SETTINGS
    ## SVM PARAMETERS GRID SEARCH
    # Define the parameters for grid search
    parameters = {
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'C':[0.1, 1, 10],
        #'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        #'degree': [2, 3, 4],
        #'coef0': [0.0, 0.5, 1.0],
        #'shrinking': [True, False],
        #'tol': [1e-3, 1e-4, 1e-5],
        'max_iter': [-1] #[-1, 100, 500, 1000, 2000]
    }

    ### UPDATE DATA
    for obj_path in obj_path_list:
        seizure_records  = os.listdir(obj_path)
        for seizure_record in seizure_records:
            if os.path.isdir(obj_path + "/" + seizure_record):
                files = os.listdir(obj_path + "/" + seizure_record)
                for file in files:

                    def load_object(filename):
                        with open(filename, 'rb') as file:
                            obj = pickle.load(file)
                        return obj 
                        
                    if file.find(patient)!=-1 or patient=="ALL":
                    
                        try: 
                            time_signal = load_object(obj_path + "/" + seizure_record + "/" + file) #EOFError: Ran out of input                        
                        except:
                            continue
                        samples, labels, obj_name = time_signal.IMs_SVM_bands(scaled=True, flag_ch=flag_channels)

                        # Concatenate n_conc windows
                        samples, labels = Signal.concatenate_windows(samples, labels, n_conc)
        
                        dataset_samples.append(samples)
                        dataset_labels.append(labels)
                        obj_names.append(obj_name)
                        print(file)
                        #dataset.append(time_signal.IMs_SVM())
                     
    # leave-one-records-out Cross Validation                                      
    for indx, name in enumerate(obj_names):

        print("obj excluded: ", name)
        print("n_conc ", str(n_conc))

        # Training samples
        X_train = np.concatenate(dataset_samples[:indx] + dataset_samples[indx+1:], axis=2)
        X_train = np.reshape(np.transpose(X_train, (2, 1, 0)), (X_train.shape[2], X_train.shape[0]*X_train.shape[1]))
        Y_train = sum(dataset_labels[:indx] + dataset_labels[indx+1:] , []) 
        #  , [])  serves to initializate the sum, and define the type of data will ve summed

        # Test samples
        X_test = dataset_samples[indx]
        X_test = np.reshape(np.transpose(X_test, (2, 1, 0)), (X_test.shape[2], X_test.shape[0]*X_test.shape[1]))
        Y_test = dataset_labels[indx]

        ### SVM
        # Define the SVM model
        svm = SVC()
        # Create the grid search object
        grid_search = GridSearchCV(svm, parameters, cv=5)
        print("executing SVM grid search...")
        grid_search.fit(X_train, Y_train)
        # Print the best parameters and the best score
        print("Best parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
        
        # Evaluate the model on the testing data using the best parameters
        svm_best = SVC(**grid_search.best_params_, probability=True)
        svm_best.fit(X_train, Y_train)
        predict = svm_best.predict(X_test)
        predict_proba = svm_best.predict_proba(X_test)
        score = svm_best.score(X_test, Y_test)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(Y_test, predict).ravel()
        # Compute sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        # Print Test results
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("Accuracy on testing data: ", score)

        # Create a plot
        fig, ax = plt.subplots()
        # Plot a horizontal line at y = 3
        ax.axhline(y=1, color='b', linestyle="--")
        #ax.plot(np.abs(dataset_samples[indx][0,:12,:]), linewidth=0.3,  color='gray', alpha=0.5)
        ax.plot(dataset_labels[indx], linewidth=1.5, color='r')
        #plt.plot(predict, linewidth=3, color='green')
        ax.plot(np.hstack([np.array([0] * n_conc), predict]), linewidth=1.5, color='green')
        ax.plot(np.hstack([np.array([0] * n_conc), predict_proba[:,1]]), linestyle="-",  alpha=0.5)
        # Show the plot
        plt.show()


        ### RANDOM FOREST
        rf = RandomForestClassifier()
        rf.fit(X_train, Y_train)
        predict = rf.predict(X_test)
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(Y_test, predict).ravel()
        # Compute sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        # Print Test results
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("Accuracy on testing data: ", score)

        # Create a plot
        fig, ax = plt.subplots()
        # Plot a horizontal line at y = 3
        ax.axhline(y=1, color='b', linestyle="--")
        #ax.plot(np.abs(dataset_samples[indx][:,:12]), linewidth=0.3,  color='gray', alpha=0.5)
        ax.plot(dataset_labels[indx], linewidth=1.5, color='r')
        #plt.plot(predict, linewidth=3, color='green')
        ax.plot(np.hstack([np.array([0] * n_conc), predict]), linewidth=1.5, color='green')
        #ax.plot(np.hstack([np.array([0] * n_conc), predict_proba[:,1]]), linestyle="-",  alpha=0.5)
        # Show the plot
        plt.show()





    dataset_samples = np.vstack(dataset_samples)
    dataset_labels = sum(dataset_labels, [])
    # In the above example, sum(list_of_lists, []) concatenates all the sublists in list_of_lists into one list. The second argument [] is the initial value of the accumulator used by sum and indicates that we want to start with an empty list.

    ### Standardization
    # We normalise each record individually
    #dataset_samples = (dataset_samples-np.mean(dataset_samples, axis=0))/np.std(dataset_samples, axis=0)
    #scaler = StandardScaler()
    #dataset_samples_scaled = scaler.fit_transform(dataset_samples)
    #X_test_scaled = scaler.transform(X_test)
    dataset_samples_scaled = dataset_samples

    ### SVM
    # Define the SVM model
    svm = SVC()

    # Define the parameters for grid search
    parameters = {
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'C':[0.1, 1, 10],
        #'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        #'degree': [2, 3, 4],
        #'coef0': [0.0, 0.5, 1.0],
        #'shrinking': [True, False],
        #'tol': [1e-3, 1e-4, 1e-5],
        'max_iter': [-1] #[-1, 100, 500, 1000, 2000]
    }

    # Create the grid search object
    grid_search = GridSearchCV(svm, parameters, cv=5)

    # Save originals
    dataset_samples_scaled_original = dataset_samples_scaled
    dataset_labels_original = dataset_labels

    # Concatenate n_conc windows
    dataset_samples_scaled, dataset_labels = Signal.concatenate_windows(dataset_samples_scaled, dataset_labels, n_conc)
    X_train, X_test, y_train, y_test = train_test_split(dataset_samples_scaled, dataset_labels, test_size=0.2)
    # N = len(dataset_labels)
    # por_train = 0.9
    # X_train = dataset_samples_scaled[:int(N*por_train)]
    # X_test = dataset_samples_scaled[int(N*por_train):]
    # y_train = dataset_labels[:int(N*por_train)]
    # y_test = dataset_labels[int(N*por_train):]


    #Test con Lazypredict
    #Initializing and training the lazy classifier
    #clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    #models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    #clf.predictions
    #Printing the results
    #print(models)

    #Predict with RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(dataset_samples_scaled)
    tn, fp, fn, tp = confusion_matrix(rf.predict(X_test), y_test).ravel()

    #plt.figure(figsize=(5, 10))
    #sns.set_theme(style="whitegrid")
    #ax = sns.barplot(y=predictions.index, x="Accuracy", data=predictions)
    # Fit the grid search object to the training data
    print("executing SVM grid search...")
    grid_search.fit(dataset_samples_scaled, dataset_labels)

    # Print the best parameters and the best score
    print("n_conc ", str(n_conc))
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    # Evaluate the model on the testing data using the best parameters
    svm_best = SVC(**grid_search.best_params_, probability=True)
    svm_best.fit(dataset_samples_scaled, dataset_labels)
    predict = svm_best.predict(dataset_samples_scaled)
    predict_proba = svm_best.predict_proba(dataset_samples_scaled)

    score = svm_best.score(dataset_samples_scaled, dataset_labels)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(dataset_labels, predict).ravel()
    # Compute sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Print the results
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    print("Accuracy on testing data: ", score)




