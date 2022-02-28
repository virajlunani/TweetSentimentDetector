import nrclex
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
