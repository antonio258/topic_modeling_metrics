import math
import logging
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance

from scipy.sparse import csr_matrix
from topic_modeling_metrics.utils import Logger


class TraditionalMetrics:
    def __init__(self, docs, topic_words, n_words=None):
        self.docs = docs
        self.topic_words = topic_words
        self.logger = Logger("traditional_metrics").get_logger()
        self.n_words = n_words
        if not n_words:
            self.n_words = len(topic_words[list(topic_words.keys())[0]])
        else:
            for topic in self.topic_words:
                self.topic_words[topic] = self.topic_words[topic][:n_words]

        self.embedding = None

    @staticmethod
    def _get_coocorrency(word_index1: int, word_index2: int, X: csr_matrix):

        count_word1 = X[:, word_index1].toarray()
        count_word2 = X[:, word_index2].toarray()

        count_concat = np.concatenate((count_word1, count_word2), axis=1)
        count_min = np.min(count_concat, axis=1)
        return count_min.sum()

    def _npmi(self, X: csr_matrix, vocab: np.ndarray, word_index1: int, word_index2: int):
        try:  # calcula o pmi
            pmi = math.log(
                (
                    self._get_coocorrency(word_index1, word_index2, X) / X.sum()
                ) /  # probabilidade de coocorrencia (total_coocoreencia/total_palavras)
                (
                    (X.getcol(word_index1).sum() / X.sum()) * (X.getcol(word_index2).sum() / X.sum())
                )  # probabilidade palavra 1 x probabilidade palavra 2. Probabilidade = total_ocorrencia/total_palavras
            )
        except ValueError:
            pmi = 0

        try:
            npmi = (
                    pmi /
                    (-1 * (math.log(self._get_coocorrency(word_index1, word_index2, X) / X.sum())))
            )
        except ValueError:
            npmi = 0
        return npmi

    def get_npmi(self):
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "), token_pattern=None)
        X = vectorizer.fit_transform(self.docs)  # tf matrix
        vocab = vectorizer.get_feature_names_out()  # vocab list

        topics_npmi = {}
        for topic_number, topic in self.topic_words.items():
            topics_results = []
            for i in range(len(topic)):
                for j in range(i + 1, len(topic)):
                    # print(topic[i], topic[j])
                    try:
                        word_index1 = np.where(vocab == topic[i])[0][0]  # indice da palavra 1
                        word_index2 = np.where(vocab == topic[j])[0][0]  # indice da palavra 2
                    except Exception as e:
                        print(repr(topic[i]), repr(topic[j]))
                        raise e
                    npmi = self._npmi(X, vocab, word_index1, word_index2)

                    topics_results.append(npmi)
            topics_npmi[topic_number] = np.mean(topics_results)#, np.std(topics_results), np.median(topics_results)

        return topics_npmi

    def _count_tf_idf_repr(self):
        vectorizer = TfidfVectorizer(encoding='utf-8',
                                     analyzer='word',
                                     use_idf=True,
                                     smooth_idf=False,
                                     stop_words=None,
                                     tokenizer=lambda x: x.split(' '),
                                     token_pattern=None)
        X = vectorizer.fit_transform(self.docs)
        tf_idf_t = csr_matrix(X).transpose()

        words = vectorizer.get_feature_names_out()
        frequency = {}
        docs = {}
        for index, topic in self.topic_words.items():
            for word in topic:
                word_index = np.where(words == word)[0]
                frequency[word] = float(tf_idf_t[word_index].data.shape[0])
                docs[word] = set(tf_idf_t[word_index].nonzero()[1])

        return frequency, docs

    def get_coherence(self):
        word_frequency, term_docs = self._count_tf_idf_repr()
        coherence = {}

        for topic_id, top_w in self.topic_words.items():

            coherence_t = 0.0
            for i in range(1, len(top_w)):
                for j in range(0, i):
                    cont_wi = word_frequency[top_w[j]]
                    cont_wi_wj = float(
                        len(term_docs[top_w[j]].intersection(term_docs[top_w[i]])))
                    coherence_t += np.log((cont_wi_wj + 1.0) / cont_wi)

            coherence[topic_id] = coherence_t

        return coherence

    def _get_word_embedding(self, word):
        try:
            return self.embedding[word].tolist()
        except KeyError:
            return None

    def get_embedding_distance(self, embedding_path: str, binary_embedding=False):
        self.embedding = KeyedVectors.load_word2vec_format(embedding_path, binary=binary_embedding)
        topic_size = sum([len(self.topic_words[x]) for x in self.topic_words])
        embedding_topic_size = 0
        topic_embedding = []
        for topic in self.topic_words:
            te = [self._get_word_embedding(x) for x in self.topic_words[topic]]
            te = [x for x in te if x]
            embedding_topic_size += len(te)
            topic_embedding.append(te[:])

        self.logger.info(f"{embedding_topic_size / topic_size} de palavras em tópicos mapeadas...")

        embedding_metrice = {}
        for topic_number, topic in enumerate(topic_embedding):
            topics_results = []
            n_words = len(topic)
            for i in range(n_words):
                for j in range(i + 1, n_words):
                    result = 1 - distance.cosine(topic_embedding[topic_number][i], topic_embedding[topic_number][j])
                    topics_results.append(result)
            embedding_metrice[topic_number] = np.mean(topics_results)

        return embedding_metrice
