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
def create_vocab(imdb_data):
    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=40000,
        max_df=.6,
        min_df=2
    )

    # fit the vectorizer on the text
    # vectorizer.fit(imdb_data['review'])

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
    # print(total_spam, spam_total)
    # print(total_non_spam, non_spam_total)

    spam_likelyhood = []
    non_spam_likelyhood = []
    alpha = 1

    # get the vocabulary
    # inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    # vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]


    for value in total_spam:
        spam_likelyhood.append(np.log((value + alpha) / (spam_total + alpha * len(vector[0]))))

    for value in total_non_spam:
        non_spam_likelyhood.append(np.log((value + alpha) / (non_spam_total + alpha * len(vector[0]))))


    pred = -1
    i = 0
    correct = 0
    pred_list = []
    vector_validate = np.copy(vector[30000:40000])
    for item in vector_validate:
        spam_value = (item.T*spam_likelyhood).T.sum() + prob[0]
        non_spam_value = (item.T*non_spam_likelyhood).T.sum() + prob[1]

        if(spam_value > non_spam_value):
            pred = "positive"
        else:
            pred = "negative"
        if(pred == valid_labs[i]):
            correct += 1
        pred_list.append(pred)
        i += 1

    return pred_list, correct

start = time.time()

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

# Get the vocabulary and two lists of ints of word occurances plus the likelyhoods
prob = [np.log(training_labs.count("negative") / len(training_labs)), np.log(training_labs.count("positive") / len(training_labs))]
pred_list, correct = create_vocab(imdb_data)

# set an alpha for Laplace
alpha = 1

# Set up the three lists we'll be using for classification
# wcList = getLabs()
# likelyhoods, prob = getLikelyhoods(alpha)

# Run the classification on the validation data
# pred_list, correct = classify(valid_data, valid_labs, likelyhoods)

# Print out the percent correct on the validation set
print("Percent Correct: " + str(round(correct/len(valid_labs)*100,2)))

# Print the list of predictions to an output file
with open('test-prediciton3.csv', 'w') as outfile:
    for line in pred_list:
        outfile.write(line + "\n")

end = time.time()

print(end-start)