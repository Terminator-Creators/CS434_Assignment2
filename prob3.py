import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import math
import time
import numpy as np

# Function to make a 1D list of sentences into a 2D list of words
def create_list(aList):
    bList = []
    for line in aList:
        line = clean_text(line)
        bList.append(line.split(" "))
    return bList


# Preproccessor for the vectorizer
def clean_text(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    return text


# Function to create the vocab from the imdb_data
def create_vocab(imdb_data, alpha, vec_lwr, vec_upr):
    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=2000,
    )

    # This bit grabs the total word counts for positive and negative reviews, should speed up calc time
    vector = vectorizer.fit_transform(imdb_data['review']).toarray()
    vector_train = np.copy(vector[:30000])
    training_labs = label_data[:30000].iloc[:, 0].tolist()
    label, original = pd.factorize(training_labs)
    subtractor = np.full(shape=30000, fill_value=1, dtype=int)
    spam_label = np.subtract(label, subtractor)
    spam_label = np.abs(spam_label)
    non_spam_count = (vector_train.T*label).T
    spam_count = (vector_train.T*spam_label).T

    total_spam = spam_count.sum(axis=0)
    total_non_spam = non_spam_count.sum(axis=0)
    
    spam_total = total_spam.sum()
    non_spam_total = total_non_spam.sum()

    spam_likelyhood = []
    non_spam_likelyhood = []


    for value in total_spam:
        spam_likelyhood.append(np.log((value + alpha) / (spam_total + alpha * len(vector[0]))))

    for value in total_non_spam:
        non_spam_likelyhood.append(np.log((value + alpha) / (non_spam_total + alpha * len(vector[0]))))


    pred = -1
    i = 0
    correct = 0
    pred_list = []
    # Setting the vec_lwr and vec_upr will let you choose which set to use:
    # vec_lwr = 30000 and vec_upr = 40000 gives validation
    # vec_lwr = 40000 and vec_upr = 50000 gives testing
    vector_validate = np.copy(vector[vec_lwr:vec_upr])
    for item in vector_validate:
        spam_value = (item.T*spam_likelyhood).T.sum() + prob[0]
        non_spam_value = (item.T*non_spam_likelyhood).T.sum() + prob[1]

        if(spam_value > non_spam_value):
            pred = "positive"
        else:
            pred = "negative"
        if(vec_lwr == 30000):
            if(pred == valid_labs[i]):
                correct += 1
        pred_list.append(pred)
        i += 1

    return pred_list, correct


# Importing the dataset and creating the three distinct datasets
imdb_data  = pd.read_csv('IMDB.csv', delimiter=',')
label_data = pd.read_csv('IMDB_labels.csv', delimiter=',') 

# Separate out the three data sets and clean them using the preprocessor
training_data = create_list(imdb_data[:30000].iloc[:, 0].tolist())
valid_data    = create_list(imdb_data[30000:40000].iloc[:, 0].tolist())
testing_data  = create_list(imdb_data[40000:].iloc[:, 0].tolist())

# Break out the validation labels from the training labels
training_labs = label_data[:30000].iloc[:, 0].tolist()
valid_labs    = label_data[30000:40000].iloc[:, 0].tolist()

# set an alpha for Laplace
alpha = 1
prob = [np.log(training_labs.count("negative") / len(training_labs)), np.log(training_labs.count("positive") / len(training_labs))]

# Get the vocabulary and two lists of ints of word occurances plus the likelyhoods
#30000 - 40000 is the valid data set
pred_list, correct = create_vocab(imdb_data, alpha, 30000, 40000)

# Print out the percent correct on the validation set
print("Percent Correct Valid Data: " + str(round(correct/len(valid_labs)*100,2)))

#40000 - 50000 is the testing data set
pred_list, correct = create_vocab(imdb_data, alpha, 40000, 50000)

# Print the list of predictions of the test set to an output file
with open('test-prediction1.csv', 'w') as outfile:
    for line in pred_list:
        outfile.write(line + "\n")