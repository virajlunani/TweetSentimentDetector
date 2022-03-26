import nrclex
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def extract_nrc_pos_feats(df, features):
    feature_freq = np.zeros((len(df), len(features)))
    for i, tweet in enumerate(df['text']):
        tokenized = nltk.word_tokenize(tweet)
        num_tokens = len(tokenized)
        if (num_tokens == 0):
            continue
        emo_scores = nrclex.NRCLex(tweet).raw_emotion_scores
        for emotion, count in emo_scores.items():
            index = features[emotion]
            feature_freq[i][index] = count
        for token in tokenized:
            if token == '?':
                index = features['?']
                feature_freq[i][index] += 1
            if token == '!':
                index = features['!']
                feature_freq[i][index] += 1
        tags = nltk.pos_tag(tokenized)
        for _, tag in tags:
            if tag == 'CC':
                feature_freq[i][10] += 1
            elif tag == 'IN': 
                feature_freq[i][11] += 1
            elif tag == 'JJR': 
                feature_freq[i][12] += 1
            elif tag == 'JJS': 
                feature_freq[i][13] += 1
            elif tag == 'PRP': 
                feature_freq[i][14] += 1
        for index in range(len(features)):
            feature_freq[i][index] /= num_tokens
            feature_freq[i][index] *= 100

    return feature_freq

def stem_text(text):
    ps = nltk.stem.PorterStemmer()
    reconstructed = []
    tweet = text['text']
    for word in tweet.split(' '):
        reconstructed.append(ps.stem(word))
    text['text'] = ' '.join(reconstructed)
    return text

def stem_data(df):
    df = df.apply(stem_text, axis=1)
    return df

def removeStop(df, stopeng):
    # tokenize
    df['text'] = df['text'].map(nltk.tokenize.word_tokenize)
    # remove stopwords
    df['text'] = df['text'].apply(lambda words: [word for word in words if word not in stopeng])
    # join sentence back together
    df['text'] = df['text'].apply(lambda words: ' '.join(words))
    return df

def cleanData(df):
    stemmed = stem_data(df)
    stopeng = set(nltk.corpus.stopwords.words('english'))
    cleaned = removeStop(stemmed, stopeng)
    return cleaned

def createBoWFeatureVec(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df.loc[:, 'label'].values
    return X, y, vectorizer

def createTfIdfFeatureVec(df):
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(df['text']).toarray()
	y = df.loc[:, 'label'].values
	return X, y, vectorizer

def trainLogisticRegression(X_train, X_test, y_train, y_test):
	# lbfgs can't converge, so using newton-cg instead #
    clf = LogisticRegression(random_state=42,multi_class='multinomial',solver='newton-cg').fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return clf, round(accuracy,3)

def trainLogisticRegression_weighted(X_train, X_test, y_train, y_test, weights):
	# lbfgs can't converge, so using newton-cg instead #
    clf = LogisticRegression(random_state=42,multi_class='multinomial',solver='newton-cg',class_weight=weights).fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return clf, round(accuracy,3)

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
	
  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict
