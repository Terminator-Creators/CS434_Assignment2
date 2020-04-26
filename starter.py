import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import math

# Function to make a 1D list of sentences into a 2D list of words
def create_list(aList):
    bList = []
    for line in aList:
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




# Importing the dataset and creating the three distinct datasets
imdb_data  = pd.read_csv('IMDB.csv', delimiter=',')
label_data = pd.read_csv('IMDB_labels.csv', delimiter=',') 

# Need a better way to to_list these, since we need the sanitized version
training_data = create_list(imdb_data[:30000].iloc[:, 0].tolist())
valid_data    = create_list(imdb_data[30000:40001].iloc[:, 0].tolist())

testing_data  = create_list(imdb_data[40001:].iloc[:, 0].tolist())

training_labs = label_data[:30000].iloc[:, 0].tolist()
valid_labs    = label_data[30000:40001].iloc[:, 0].tolist()

# Get the vocabulary
vocabulary = create_vocab(imdb_data)

# set an alpha for Laplace
alpha = 1

# Set up the two lists we'll be filling in this for loop
likelyhoods = []
mylist = [[],[]]
prob = []

# Go through each review and add each word to a list depending if the review is positive of negative
for i, lab in enumerate(set(training_labs)):
    prob.append(math.log(training_labs.count(lab) / len(training_labs)))
    for j, elem in enumerate(training_labs):
        line = training_data[j]
        # Only two classes here, so we can shortcut
        if(elem == "positive"):
            for word in line:
                if(word in vocabulary):
                    mylist[1].append(word)
        else:
            for word in line:
                if(word in vocabulary):
                    mylist[0].append(word)



        # Get the total number of words from the vocabulary
        total_words = 0
        for word in vocabulary:
            total_words += mylist[i].count(word)

        # Compute the likelyhoods of each word in the vocabulary
        likelyhoods.clear()
        for word in vocabulary:
            count = mylist.count(word)
            likelyhoods.append(math.log((count + alpha) / (total_words + alpha * len(vocabulary))))

# For each review, grab the likelyhood that word is in a positive review and sum them
# Not sure on this part, definitely need to understand this better
i = -1
correct = 0
pred_list = []

for line in valid_data:
    i += 1
    sums = [0]*2
    for j, lab in enumerate(set(training_labs)):
        sums[j] = prob[j]
        for word in line:
            if word in vocabulary:
                sums[j] += likelyhoods[likelyhoods.index(word)]
    
    pred = -1
    if(sums[0] >= sums[1]):
        pred = "negative"
    else:
        pred = "positive"

    if(pred == valid_labs[i]):
        correct += 1
    pred_list.append(pred)
    
print("Percent Correct: " + str(correct/len(valid_labs)) + "\n\n")
print(pred_list)