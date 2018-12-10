from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB

cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words
             if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token),
                       words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token:
                                  p.match(token) and len(token) >= min_length,
                                  tokens))
    return filtered_tokens


# Return the representer, without transforming
def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3,
                            max_df=0.90, max_features=3000,
                            use_idf=True, sublinear_tf=True,
                            norm='l2')
    tfidf.fit(docs)
    return tfidf


def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return [(features[index], doc_representation[0, index])
            for index in doc_representation.nonzero()[1]]


if __name__ == '__main__':
    train_docs = []
    train_labels = []
    test_docs = []
    test_labels = []

    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
            train_labels.append(reuters.categories(doc_id)[0])
        else:
            test_docs.append(reuters.raw(doc_id))
            test_labels.append(reuters.categories(doc_id))

    representer = tf_idf(train_docs)
    classifier = GaussianNB()
    classifier.fit(representer.fit_transform(train_docs).toarray(), train_labels)
    representer = tf_idf(test_docs)
    prediction = classifier.predict(representer.fit_transform(test_docs).toarray())

    g = 0
    b = 0
    for i, c in enumerate(prediction):
        if c in prediction[0]:
            g += 1
        else:
            b += 1

    print(g/(g+b))

    # print(accuracy_score(test_labels, prediction))
    # print(confusion_matrix(test_labels, prediction))
    # print(classification_report(test_labels, prediction))
    # print(representer.transform(train_docs))
    # for doc in test_docs:
    #     print(feature_values(doc, representer))
