import pandas
from math import sqrt
from math import exp
from math import pi
from math import isnan

# Load xlsx file
def load_xlsx(filename):
    xlsx_file = pandas.read_excel(filename)
    samples = []
    # Convert "L" and "R" to 0 and 1
    for row in xlsx_file.values:
        if row[3] == "L":
            row[3] = 0
        else:
            row[3] = 1
        samples.append(list(row[2:]))
    # Convert nan to 0
    for row in samples:
        for index in range(len(row)):
            if isnan(row[index]):
                row[index] = 0
    labels = xlsx_file["Target"].values
    return samples, labels

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset, labels):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = labels[i]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 
# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	# del(summaries[-1])
	return summaries
 
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset, labels):
	separated = separate_by_class(dataset, labels)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries
 
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    if stdev == 0:
        stdev = 0.0001
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent
 
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities
 
# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label



# Make a prediction with Naive Bayes on Iris Dataset
samples, labels = load_xlsx("Project_NB_Tr.xlsx")
# fit model
model = summarize_by_class(samples, labels)

accuracy = 0
sample_num = 0
for row, gt in zip(samples, labels):
    sample_num += 1
    label = predict(model, row)
    if gt == label:
        accuracy += 1
    # print('Data=%s, Predicted: %s' % (row, label))
print("Test accuracy: ", accuracy/sample_num)