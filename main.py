from typing import Tuple, Any, List

from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def preprocess(text: str) -> str:
    """
    This method cleans text using nltk.lemmatizer
    :param text: input text
    :return: cleaned text
    """
    # Remove Punctuation
    messages_cleaned = re.sub(r'[^\w\s]', '', text)

    # Remove Links
    messages_cleaned = re.sub(r'https?://\S+', '', messages_cleaned)

    # Lower Case
    messages_cleaned = messages_cleaned.lower()

    # Tokenize
    messages_cleaned = word_tokenize(messages_cleaned)

    # Remove Stopwords
    stop_words_list = stopwords.words('english')
    for token in messages_cleaned:
        if token in stop_words_list:
            messages_cleaned.remove(token)
    # lemmatizing text
    lemmatizer = WordNetLemmatizer()
    temp = messages_cleaned.copy()
    messages_cleaned.clear()
    for token in temp:
        messages_cleaned.append(lemmatizer.lemmatize(token))

    # rejoin message tokens
    messages_cleaned = ' '.join(messages_cleaned)
    return messages_cleaned


class LatentDirichletAllocation(TransformerMixin):
    """
    This class represents a custom implementation of LDA.
    """

    def __init__(self, k: int, vocab: np.ndarray, iters: int) -> None:
        """
        Initialise the class params
        :param k: number of topics
        :param vocab: total number of unique words present in data
        :param iters: number of iterations
        """
        self.k = k
        self.vocab = vocab
        self.iters = iters
    def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> Tuple[Any, Any]:
        """
        Main implementation of projecting vocab into topics of size k.
        :param X: data
        :param y:
        :param fit_params:
        :return: doc_topic_dist and topic_word_dist
        """
        # Get DLMax Matrix
        docs = []
        for row in X:
            word_indices = np.where(row != 0)[0].tolist()  # indices where word count > 0
            word_counts = []
            for word_idx in word_indices:
                count = row[word_idx]
                for cnt in range(count):
                    word_counts.append(word_idx)

            docs.append(word_counts)

        N = len(docs)  # number of documents
        W = len(self.vocab)  # size of the vocabulary
        alpha = 5 * np.ones(self.k)  # Dirichlet prior of the document-topic distribution
        beta = 2 * np.ones(W)  # Dirichlet prior of the topic-word distribution

        Z = [[0 for _ in range(len(d))] for d in docs]
        A = np.zeros(shape=(N, self.k))
        B = np.zeros(shape=(self.k, W))
        n_d = np.zeros(N)
        BSUM = np.zeros(self.k)

        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                # assign a topic randomly to words
                Z[d][i] = i % self.k
                # get the topic for word n in document m
                z = Z[d][i]
                # keep track of our counts
                A[d][z] += 1
                B[z][w] += 1
                BSUM[z] += 1
                n_d[d] += 1

        for iteration in range(self.iters):
            for d, doc in enumerate(docs):
                for i, w in enumerate(doc):
                    z_i = Z[d][i]

                    A[d, z_i] = A[d, z_i] - 1
                    B[z_i, w] = B[z_i, w] - 1
                    BSUM[z_i] = BSUM[z_i] - 1

                    p_d_t = (A[d] + alpha) / (n_d[d] - 1 + W * alpha)
                    p_t_w = (B[:, w] + beta[w]) / (BSUM + W * beta[w])
                    p_z = p_d_t * p_t_w
                    p_z = p_z / np.sum(p_z)

                    new_z_i = np.random.multinomial(1, p_z).argmax()

                    Z[d][i] = new_z_i
                    A[d, new_z_i] = A[d, new_z_i] + 1
                    B[new_z_i, w] = B[new_z_i, w] + 1
                    BSUM[new_z_i] = BSUM[new_z_i] + 1

        return A, B

    def get_top_words(self, topic_word_dist: List[Any], n_top_words: int = 20) -> pd.DataFrame:
        """
        This function returns the top words present in the document.
        :param topic_word_dist:  topic to word distribution
        :param n_top_words: number of top words
        :return:
        """
        df = {}
        for i, component in enumerate(topic_word_dist):
            tot_score = sum(component)
            topic_word_probs = np.array([j / tot_score for j in component])
            top_words_idx = np.argsort(-topic_word_probs)[:n_top_words]
            top_words = self.vocab[top_words_idx]
            top_probs = topic_word_probs[top_words_idx]
            word_w_prob = [top_words[k] + '(' + str(round(top_probs[k], 4)) + ')' for k in range(len(top_words))]
            df['Topic' + str(i)] = word_w_prob

        return pd.DataFrame(df)


if __name__ == "__main__":

    # articles = []
    # with open('sonnet.txt', 'r') as f:
    #     for line in f:
    #         articles.append(line)
    #
    # count_vect = CountVectorizer(min_df=2, max_df=0.95, max_features=10000, stop_words='english')
    # X_train = count_vect.fit_transform(articles).toarray()
    # vocab = np.array(count_vect.get_feature_names_out())
    #
    # lda = LatentDirichletAllocation(6, vocab, 100)
    #
    # doc_topic_dist, topic_word_dist = lda.fit_transform(X_train)
    #
    # df = lda.get_top_words(topic_word_dist, n_top_words=10)
    # # plotting word clouds
    #
    # for i in range(6):
    #     d = {}
    #     for j in range(len(vocab)):
    #         d[vocab[j]] = topic_word_dist[i][j]
    #
    #     wordcloud = WordCloud()
    #     wordcloud.generate_from_frequencies(frequencies=d)
    #     plt.figure()
    #     plt.imshow(wordcloud, interpolation="bilinear")
    #     plt.axis("off")
    #     plt.show()

    # X Data
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # Y Data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    X_train_org = []
    for words in newsgroups_train.data:
        X_train_org.append(preprocess(words))

    # Clean X_test

    X_test_org = []
    for words in newsgroups_test.data:
        X_test_org.append(preprocess(words))

    count_vect = CountVectorizer(max_df=0.95, min_df=2, max_features=53000, stop_words=stopwords.words('english'))
    X_train = count_vect.fit_transform(X_train_org).toarray()
    vocab = np.array(count_vect.get_feature_names_out())

    lda = LatentDirichletAllocation(k=6, vocab=vocab, iters=10)
    doc_topic_dist, topic_word_dist = lda.fit_transform(X_train)

    for i in range(6):
        d = {}
        for j in range(len(vocab)):
            d[vocab[j]] = topic_word_dist[i][j]

        wordcloud = WordCloud()
        wordcloud.generate_from_frequencies(frequencies=d)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
