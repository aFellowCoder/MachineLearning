from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
import time

# 0 = sneakers
# 1 = ankle boots
# David Irwin
# Call Task 3,4,5 & 6 separately
# Note there are many prints in this assignment


def Task1():
    dataset = pd.read_csv('product_images.csv')

    # Split the data into label and features
    datasetLabels = dataset['label']
    datasetFeatures = dataset.drop(['label'], axis=1)

    # I then get the first instance of ankle boot and the first instance of sneaker.
    # I use iloc to start at column 1 to avoid the label column.
    ankleBootDatasetImage = dataset[dataset['label'] == 1].iloc[0, 1:].values
    sneakerDatasetImage = dataset[dataset['label'] == 0].iloc[0, 1:].values

    # print the total number of rows where the label is sneaker and where the label is ankle boot.
    print("Total samples of sneakers: ", len(dataset[dataset['label'] == 0]))
    print("Total samples of ankle boots: ", len(dataset[dataset['label'] == 1]), "\n")

    # I reshape the image data to a 28x28 array and print it using imshow.
    plt.imshow(ankleBootDatasetImage.reshape(28, 28), cmap='gray')
    plt.show()
    plt.imshow(sneakerDatasetImage.reshape(28, 28), cmap='gray')
    plt.show()

    return datasetLabels, datasetFeatures


# The modelParameter value for Task2 is the value of the parameter in the SVM model / kNN model / decision tree.
# A value of -1 being passed in is to signify that the selected model will not be using this parameterValue.
def Task2(datasetLabels, datasetFeatures, numberOfSamples, modelType, modelParameter):

    # I was unsure if we had to create training / test data & then further split the training data into training /
    # test data. So I decided to do it. This way I will be able to calculate model accuracy on unseen data.

    test_data_size = 0.2
    # Split the data into training and testing sets (The test data & test target is the unseen data)
    # I use a random state of 50 to ensure that the data is split in the same way every time so the results between
    # the models are more comparable.
    train_data, test_data, train_target, test_target = model_selection.train_test_split(datasetFeatures, datasetLabels,
                                                                        test_size=test_data_size, random_state=50)
    # Create splits for training and testing
    kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=50)

    # Parameterizing the samples size of the dataset. I.E I am reducing the size of train data & train target datasets
    # to the numberOfSamples variable. The samples are chosen at random.
    # Maximum value of numberOfSamples is len(train_data) which is 11200. (This is because the rest of the data is held
    # back as unseen data
    train_data = train_data.sample(n=numberOfSamples, random_state=50)
    train_target = train_target.sample(n=numberOfSamples, random_state=50)



    # Convert the dataframes to numpy arrays
    train_data = train_data.values
    train_target = train_target.values
    test_data = test_data.values
    test_target = test_target.values

    # Keeps track of the current split
    iteration = 1

    # Holds the training time for each split
    training_time_values = []
    # Holds the testing time for each split
    testing_time_values = []
    # Holds the accuracy for each split
    accuracy_values = []



    for train_index, test_index in kf.split(train_data):

        # Determining the model to use
        if modelType == "Perceptron":
            clf1 = linear_model.Perceptron()
        elif modelType == "SVM":
            clf1 = svm.SVC(kernel='rbf', gamma=modelParameter)
        elif modelType == "kNN":
            clf1 = neighbors.KNeighborsClassifier(n_neighbors=modelParameter)
        elif modelType == "tree":
            clf1 = tree.DecisionTreeClassifier(max_depth=modelParameter)


        print("Split: ", iteration)
        start_time_training = time.time()
        clf1.fit(train_data[train_index], train_target[train_index])
        total_time_training = time.time() - start_time_training
        print("Training time:\t\t\t", total_time_training)

        start_time_prediction = time.time()
        y_pred = clf1.predict(train_data[test_index])
        total_time_prediction = time.time() - start_time_prediction
        print("Prediction time:\t\t", total_time_prediction)

        confusion_matrix = metrics.confusion_matrix(train_target[test_index], y_pred)
        score = metrics.accuracy_score(train_target[test_index], y_pred)

        print("Confusion matrix: \n", confusion_matrix)
        print(modelType, "accuracy score: ", score, '\n')

        # Add the training time, testing time, and accuracy to their lists
        training_time_values.append(total_time_training)
        testing_time_values.append(total_time_prediction)
        accuracy_values.append(score)

        iteration += 1


    # Print the average/minimum/maximum training time, testing time, and accuracy
    print("Minimum training time:\t", min(training_time_values))
    print("Minimum testing time:\t", min(testing_time_values))
    print("Minimum accuracy:\t\t", min(accuracy_values))

    print("Maximum training time:\t", max(training_time_values))
    print("Maximum testing time:\t", max(testing_time_values))
    print("Maximum accuracy:\t\t", max(accuracy_values))

    print("Average training time:\t", sum(training_time_values)/len(training_time_values))
    print("Average testing time:\t", sum(testing_time_values)/len(testing_time_values))
    print("Average accuracy:\t\t", sum(accuracy_values)/len(accuracy_values))

    # Get the prediction values of the unseen test data
    prediction = clf1.predict(test_data)

    # Counter for correct predictions on unseen data
    correct = 0

    # loop to keep track of the correct predictions
    for i in range(0, len(test_target)):
        if prediction[i] == test_target[i]:
            correct += 1

    ans = correct / len(test_target)
    print("\nAccuracy (on unseen data): ", ans, '\n')

    # I return training time, testing time, and accuracy average values. (To be used for plotting)
    return sum(training_time_values)/len(training_time_values), sum(testing_time_values)/len(testing_time_values),\
           (sum(accuracy_values)/len(accuracy_values))


def Task3():
    # Below are arrays to hold the averages of the training time, testing time, and accuracy for each model that were
    # returned by Task2
    training_time_per_sample_size = []
    prediction_time_per_sample_size = []
    average_accuracy_per_sample_size = []
    # Keeps track of sample size for plotting
    sampleSizes = []

    # Here I Loop to run the model for different sample sizes.
    # I increment the sample size by 1120 each time for faster execution as opposed to increasing by 10 / 100 each time.
    # This allows me to run the model for a total of 10 different sample sizes.
    # Each sample increases by 1120.
    sampleSize = 1120
    for i in range(0, 10):
        # Remember that the -1 parameter below is because Perceptron is not using a parameter value
        print('*' * 40, '\n', "Sample Size: ", sampleSize)
        training_time_value_average, testing_time_value_average, accuracy_average = Task2(datasetLabels,
                                                                                          datasetFeatures, sampleSize,
                                                                                          "Perceptron", -1)

        # I am adding up all the averages for the splits and appending them to the respective list.
        training_time_per_sample_size.append(training_time_value_average)
        prediction_time_per_sample_size.append(testing_time_value_average)
        average_accuracy_per_sample_size.append(accuracy_average)
        sampleSizes.append(sampleSize)

        sampleSize += 1120

    # Below I plot the average training time (per sample) vs sample size
    plt.plot(training_time_per_sample_size, sampleSizes)
    plt.xlabel("Average Training Time (s)")
    plt.ylabel("Sample Size")
    plt.title("Average Training Time vs Sample Size (Perceptron)")
    plt.show()

    # Below I plot the average prediction time (per sample) vs sample size
    plt.plot(prediction_time_per_sample_size, sampleSizes)
    plt.xlabel("Average Prediction Time (s)")
    plt.ylabel("Sample Size")
    plt.title("Average Prediction Time vs Sample Size (Perceptron)")
    plt.show()

    # Below I plot the average accuracy (per sample) vs sample size. These accuracy values are on seen data.
    plt.plot(sampleSizes, average_accuracy_per_sample_size)
    plt.xlabel("Sample Size")
    plt.ylabel("Average Accuracy (on seen data)")
    plt.title("Average Accuracy vs Sample Size (Perceptron)")
    plt.show()


def Task4():

    # Below I call Task2 and vary the gamma value to determine a value that gives high accuracy. I will keep the sample
    # size of the dataset constant. I determined the best value for gamma by the highest mean accuracy variable below
    # Note that this mean accuracy score is the mean accuracy of all the splits (in other words it is on the seen data).
    currentgamma = 0.00000000005
    highestMeanAccuracy = 0
    bestGamma = 0
    for i in range(0, 5):
        training_time_values, testing_time_values, meanAccuracy = Task2(datasetLabels, datasetFeatures, 3000, "SVM", currentgamma)
        if meanAccuracy > highestMeanAccuracy:
            highestMeanAccuracy = meanAccuracy
            bestGamma = currentgamma

        # Below I am moving the value one step closer to the decimal point
        currentgamma *= 10

    print("Best gamma value: ", bestGamma, '\n')

    # Below are arrays to hold the averages of the training time, testing time, and accuracy for each model that were
    # returned by Task2
    training_time_per_sample_size = []
    prediction_time_per_sample_size = []
    average_accuracy_per_sample_size = []

    # Keeps track of sample size for plotting
    sampleSizes = []

    # Here I Loop to run the model for different sample sizes. I use the best gamma value I determined above.
    # I increment the sample size by 1120 each time for faster execution as opposed to increasing by 10 / 100 each time.
    # This allows me to run the model for a total of 10 different sample sizes.
    # Each sample increases by 1120.
    sampleSize = 1120
    for i in range(0, 10):
        print('*' * 40, '\n', "Sample Size: ", sampleSize)
        training_time_value_average, testing_time_value_average, accuracy_average = Task2(datasetLabels,
                                                                                          datasetFeatures, sampleSize,
                                                                                          "SVM", bestGamma)

        # I am adding up all the averages for the splits and appending them to the respective list.
        training_time_per_sample_size.append(training_time_value_average)
        prediction_time_per_sample_size.append(testing_time_value_average)
        average_accuracy_per_sample_size.append(accuracy_average)
        sampleSizes.append(sampleSize)

        sampleSize += 1120

        # Below I plot the average training time (per sample) vs sample size
    plt.plot(training_time_per_sample_size, sampleSizes)
    plt.xlabel("Average Training Time (s)")
    plt.ylabel("Sample Size")
    plt.title("Average Training Time vs Sample Size (SVM)")
    plt.show()

    # Below I plot the average prediction time (per sample) vs sample size
    plt.plot(prediction_time_per_sample_size, sampleSizes)
    plt.xlabel("Average Prediction Time (s)")
    plt.ylabel("Sample Size")
    plt.title("Average Prediction Time vs Sample Size (SVM)")
    plt.show()

    # Below I plot the average accuracy (per sample) vs sample size. These accuracy values are on seen data.
    plt.plot(sampleSizes, average_accuracy_per_sample_size)
    plt.xlabel("Sample Size")
    plt.ylabel("Average Accuracy (on seen data)")
    plt.title("Average Accuracy vs Sample Size (SVM)")
    plt.show()


def Task5():
    # Below I call Task2 and vary the gamma value to determine a vale that gives high accuracy. I will keep the sample
    # size of the dataset constant. I determined the best value for k by the highest mean accuracy variable below
    # Note that this mean accuracy score is the mean accuracy of all the splits (in other words it is on the seen data).
    current_k_Value = 1
    highestMeanAccuracy = 0
    best_k_value = 0
    for i in range(0, 5):
        training_time_values, testing_time_values, meanAccuracy = Task2(datasetLabels, datasetFeatures, 3000, "kNN", current_k_Value)
        if meanAccuracy > highestMeanAccuracy:
            highestMeanAccuracy = meanAccuracy
            best_k_value = current_k_Value

        # Below I am moving the value one step closer to the decimal point
        current_k_Value += 2

    print("Best k value: ", best_k_value, '\n')

    # Below are arrays to hold the averages of the training time, testing time, and accuracy for each model that were
    # returned by Task2
    training_time_per_sample_size = []
    prediction_time_per_sample_size = []
    average_accuracy_per_sample_size = []

    # Keeps track of sample size for plotting
    sampleSizes = []

    # Here I Loop to run the model for different sample sizes. I use the best k value I determined above.
    # I increment the sample size by 1120 each time for faster execution as opposed to increasing by 10 / 100 each time.
    # This allows me to run the model for a total of 10 different sample sizes.
    # Each sample increases by 1120.
    sampleSize = 1120
    for i in range(0, 10):

        print('*' * 40, '\n', "Sample Size: ", sampleSize)
        training_time_value_average, testing_time_value_average, accuracy_average = Task2(datasetLabels,
                                                                                          datasetFeatures, sampleSize,
                                                                                          "kNN", best_k_value)

        # I am adding up all the averages for the splits and appending them to the respective list.
        training_time_per_sample_size.append(training_time_value_average)
        prediction_time_per_sample_size.append(testing_time_value_average)
        average_accuracy_per_sample_size.append(accuracy_average)
        sampleSizes.append(sampleSize)
        sampleSize += 1120

        # Below I plot the average training time (per sample) vs sample size
    plt.plot(training_time_per_sample_size, sampleSizes)
    plt.xlabel("Average Training Time (s)")
    plt.ylabel("Sample Size")
    plt.title("Average Training Time vs Sample Size (kNN)")
    plt.show()

    # Below I plot the average prediction time (per sample) vs sample size
    plt.plot(prediction_time_per_sample_size, sampleSizes)
    plt.xlabel("Average Prediction Time (s)")
    plt.ylabel("Sample Size")
    plt.title("Average Prediction Time vs Sample Size (kNN)")
    plt.show()

    # Below I plot the average accuracy (per sample) vs sample size. These accuracy values are on seen data.
    plt.plot(sampleSizes, average_accuracy_per_sample_size)
    plt.xlabel("Sample Size")
    plt.ylabel("Average Accuracy (on seen data)")
    plt.title("Average Accuracy vs Sample Size (kNN)")
    plt.show()


def Task6():
    # Below I call Task2 and vary the d value (tree depth) to determine a value that gives high accuracy. I will keep the sample
    # size of the dataset constant. I determined the best value for d by the highest mean accuracy variable below
    # Note that this mean accuracy score is the mean accuracy of all the splits (in other words it is on the seen data).
    current_d_Value = 1
    highestMeanAccuracy = 0
    best_d_value = 0
    for i in range(0, 10):
        training_time_values, testing_time_values, meanAccuracy = Task2(datasetLabels, datasetFeatures, 3000, "tree", current_d_Value)
        if meanAccuracy > highestMeanAccuracy:
            highestMeanAccuracy = meanAccuracy
            best_d_value = current_d_Value

        # Below I am moving the value one step closer to the decimal point
        current_d_Value += 1

    print("Best d value: ", best_d_value, '\n')

    # Below are arrays to hold the averages of the training time, testing time, and accuracy for each model that were
    # returned by Task2
    training_time_per_sample_size = []
    prediction_time_per_sample_size = []
    average_accuracy_per_sample_size = []

    # Keeps track of sample size for plotting
    sampleSizes = []

    # Here I Loop to run the model for different sample sizes. I use the best d value (tree depth) I determined above.
    # I increment the sample size by 1120 each time for faster execution as opposed to increasing by 10 / 100 each time.
    # This allows me to run the model for a total of 10 different sample sizes.
    # Each sample increases by 1120.
    sampleSize = 1120
    for i in range(0, 10):

        print('*' * 40, '\n', "Sample Size: ", sampleSize)
        training_time_value_average, testing_time_value_average, accuracy_average = Task2(datasetLabels,
                                                                                          datasetFeatures, sampleSize,
                                                                                          "tree", best_d_value)

        # I am adding up all the averages for the splits and appending them to the respective list.
        training_time_per_sample_size.append(training_time_value_average)
        prediction_time_per_sample_size.append(testing_time_value_average)
        average_accuracy_per_sample_size.append(accuracy_average)
        sampleSizes.append(sampleSize)
        sampleSize += 1120

        # Below I plot the average training time (per sample) vs sample size
    plt.plot(training_time_per_sample_size, sampleSizes)
    plt.xlabel("Average Training Time (s)")
    plt.ylabel("Sample Size")
    plt.title("Average Training Time vs Sample Size (Decision Tree)")
    plt.show()

    # Below I plot the average prediction time (per sample) vs sample size
    plt.plot(prediction_time_per_sample_size, sampleSizes)
    plt.xlabel("Average Prediction Time (s)")
    plt.ylabel("Sample Size")
    plt.title("Average Prediction Time vs Sample Size (Decision Tree)")
    plt.show()

    # Below I plot the average accuracy (per sample) vs sample size. These accuracy values are on seen data.
    plt.plot(sampleSizes, average_accuracy_per_sample_size)
    plt.xlabel("Sample Size")
    plt.ylabel("Average Accuracy (on seen data)")
    plt.title("Average Accuracy vs Sample Size (Decision Tree)")
    plt.show()


# Calling task 1 to get the labels and features.
datasetLabels, datasetFeatures = Task1()

# Calling model task
Task3()
