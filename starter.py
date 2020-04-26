import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import math
import time

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
        max_features=2000,
    )

    # fit the vectorizer on the text
    vectorizer.fit(imdb_data['review'])

    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    return vocabulary


# Function to propogate the two wordcount lists for each label
def getLabs():
    wcList = [[],[]]
    for j, elem in enumerate(training_labs):
        line = training_data[j]
        # Only two classes here, so we can shortcut
        if(elem == "positive"):
            for word in line:
                if(word in vocabulary):
                    wcList[1].append(word)
        else:
            for word in line:
                if(word in vocabulary):
                    wcList[0].append(word)
    return wcList


def getLikelyhoods(alpha):
    likelyhoods = [[],[]]
    prob = []
    # Loop through each label (pos and neg) and get the probability, the total words, and the likelyhoods for each word seen in that label
    for i, lab in enumerate(set(training_labs)):
        prob.append(math.log(training_labs.count(lab) / len(training_labs)))

        # Get the total number of words from the vocabulary
        total_words = 0
        for word in vocabulary:
            total_words += wcList[i].count(word)

        # Compute the likelyhoods of each word in the vocabulary
        for word in vocabulary:
            count = wcList[i].count(word)
            likelyhoods[i].append(math.log((count + alpha) / (total_words + alpha * len(vocabulary))))
    return likelyhoods, prob


def classify(valid_data, valid_labs, likelyhoods):
    i = -1
    correct = 0
    pred_list = []

    for line in valid_data:
        i += 1
        sums = [0,0]
        for j, lab in enumerate(set(valid_labs)):
            sums[j] = prob[j]
            for word in line:
                if word in vocabulary:
                    sums[j] += likelyhoods[j][vocabulary.index(word)]
        
        pred = -1
        if(sums[0] >= sums[1]):
            pred = "negative"
        else:
            pred = "positive"

        if(pred == valid_labs[i]):
            correct += 1
        pred_list.append(pred)
    return pred_list, correct


# Importing the dataset and creating the three distinct datasets
imdb_data  = pd.read_csv('IMDB.csv', delimiter=',')
label_data = pd.read_csv('IMDB_labels.csv', delimiter=',') 

# Get the vocabulary
vocabulary = create_vocab(imdb_data)

# Separate out the three data sets and clean them using the preprocessor
training_data = create_list(imdb_data[:30000].iloc[:, 0].tolist())
valid_data    = create_list(imdb_data[30000:40000].iloc[:, 0].tolist())
testing_data  = create_list(imdb_data[40000:].iloc[:, 0].tolist())

# Break out the validation labels from the training labels
training_labs = label_data[:30000].iloc[:, 0].tolist()
valid_labs    = label_data[30000:40000].iloc[:, 0].tolist()

# set an alpha for Laplace
alpha = 1

# Set up the three lists we'll be using for classification
wcList = getLabs()
likelyhoods, prob = getLikelyhoods(alpha)

# Run the classification on the validation data
pred_list, correct = classify(valid_data, valid_labs, likelyhoods)

# Print out the percent correct on the validation set
print("Percent Correct: " + str(round(correct/len(valid_labs)*100),2))

# Print the list of predictions to an output file
with open('test_predicitons.csv', 'w') as outfile:
    for line in pred_list:
        outfile.write(line + "\n")