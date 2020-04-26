import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

# Function to make a 1D list of sentences into a 2D list of words
def create_list(aList):
    return [i for item in aList for i in item.split()]

# Importing the dataset and creating the three distinct datasets
imdb_data  = pd.read_csv('IMDB.csv', delimiter=',')
label_data = pd.read_csv('IMDB_labels.csv', delimiter=',') 

# Need a better way to to_list these, since we need the sanitized version
training_data = create_list(imdb_data[:30000].iloc[:, 0].tolist())
valid_data    = create_list(imdb_data[30000:40000].iloc[:, 0].tolist())
testing_data  = create_list(imdb_data[40000:].iloc[:, 0].tolist())

training_labs = label_data[:30000].iloc[:, 0].tolist()
valid_labs    = label_data[30000:40000].iloc[:, 0].tolist()

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
# print(vocabulary)

# Grab the positive and negative probabilites based on training data
posProb = training_labs.count("positive") / len(training_labs)
negProb = 1 - posProb

# Go through each positive review and add each word to a list
mylist = []
for i, elem in enumerate(training_labs):
    if(elem == "positive"):
        line = training_data[i]
        for word in line:
            if(word in vocabulary):
                mylist.append(word)

# set an alpha for Laplace
alpha = 1

# Get the total number of words from the vocabulary
# Maybe find another way to do this too, since we have the word counts in inv_vocab
total_words = 0
for word in vocabulary:
    total_words += mylist.count(word)

# Compute the likelyhoods of each word in the vocabulary
likelyhoods = []
for word in vocabulary:
    count = mylist.count(word)
    likelyhoods.append((count + alpha) / (total_words + alpha * len(vocabulary)))

# Take the probability of a positive review
posSum = posProb

# For each review, grab the likelyhood that word is in a positive review and sum them
# Not sure on this part, definitely need to understand this better
for i, elem in enumerate(valid_data):
    word = elem
    for item in enumerate(word):
        if item in vocabulary:
            posSum += likelyhood[likelyhood.index(item)]

print(posSum)

