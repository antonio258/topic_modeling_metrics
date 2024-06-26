import numpy as np
import umap
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.autonotebook import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

from topic_modeling_metrics.utils import Logger


class ClusterMetrics:
    def __init__(self, docs, labels):
        self.topics_centroid = None
        self.logger = Logger("cluster_metrics").get_logger()
        self.docs = docs
        self.labels = labels
        self.docs_embeddings = None

    def build_embedding(self, embedding_name_or_path: str = "", embedding_method: str = "keybert",
                        embedding_binary: bool = False):
        if embedding_method == "keybert":
            if not embedding_name_or_path:
                raise ValueError("Keybert embedding requires a model name or path")
            self.logger.debug(f"keybert embeddings")
            sentence_model = SentenceTransformer(embedding_name_or_path, device="cuda")
            self.logger.debug(f"model loaded")
            embeddings = sentence_model.encode(self.docs)

        elif embedding_method == "fasttext":
            remove_count = 0
            if not embedding_name_or_path:
                raise ValueError("Fasttext embedding requires a model path")
            self.logger.debug(f"fasttext embeddings")
            embedding_model = KeyedVectors.load_word2vec_format(embedding_name_or_path, binary=embedding_binary)
            self.logger.debug(f"model loaded")
            embeddings = []

            for index, document in tqdm(self.docs):
                document_model = []
                vocab_count = 0
                for word in document.split():
                    try:
                        document_model.append(embedding_model[word])
                        vocab_count += 1
                    except KeyError:
                        pass
                if vocab_count == 0:
                    remove_count += 1
                else:
                    document_model = np.sum(document_model, axis=0) / vocab_count
                    embeddings.append(document_model.copy())
            self.logger.warning(f"Removed {remove_count} of {len(self.docs)} documents")
        elif embedding_method == "tfidf":
            self.logger.debug(f"tfidf embeddings")
            tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "))
            tfidf_embedding = tfidf_vectorizer.fit_transform(self.docs).toarray()
            embeddings = umap.UMAP(
                n_neighbors=30,
                min_dist=0.0,
                n_components=768,
                random_state=42,
            ).fit_transform(tfidf_embedding)
            self.logger.debug(f"model loaded")
        self.docs_embeddings = embeddings

    def silhouette(self):
        return silhouette_score(np.asarray(self.docs_embeddings), self.labels, metric="cosine")

    def calinski(self):
        return calinski_harabasz_score(np.asarray(self.docs_embeddings), self.labels)

    @staticmethod
    def _intra_cluster_distance(distances_row: np.ndarray, labels: np.ndarray, i: int):
        mask = labels == labels[i]
        mask[i] = False
        if not np.any(mask):
            # cluster of size 1
            return 0
        a = np.sum(distances_row[mask])
        return a

    @staticmethod
    def _inter_cluster_distance(distances_row: np.ndarray, labels: np.ndarray, i: int):
        mask = labels != labels[i]
        b = np.sum(distances_row[mask])
        return b

    @staticmethod
    def _member_count(labels: np.ndarray, i: int):
        mask = labels == i
        return len(labels[mask])

    def beta_cv(self):
        X = np.asarray(self.docs_embeddings)
        labels = np.array(self.labels)
        distances = pairwise_distances(X)
        n = labels.shape[0]
        A = np.array([self._intra_cluster_distance(distances[i], labels, i)
                      for i in range(n)])
        B = np.array([self._inter_cluster_distance(distances[i], labels, i)
                      for i in range(n)])
        a = np.sum(A)
        b = np.sum(B)
        labels_unq = np.unique(labels)
        members = np.array([self._member_count(labels, i) for i in labels_unq])
        N_in = np.array([i*(i-1) for i in members])
        n_in = np.sum(N_in)
        N_out = np.array([i*(n-i) for i in members])
        n_out = np.sum(N_out)
        betacv = (a/n_in)/(b/n_out)

        return betacv

    @staticmethod
    def _plot_2d(component1: list | np.ndarray, component2: list | np.ndarray, labels: list | np.ndarray):

        fig = go.Figure(data=go.Scatter(
            x=component1,
            y=component2,
            mode='markers',
            marker=dict(
                size=20,
                color=labels,  # set color equal to a variable
                colorscale='Rainbow',  # one of plotly colorscales
            )
        ))
        fig.update_layout(margin=dict(l=100, r=100, b=100, t=100), width=2000, height=1200)
        fig.layout.template = 'plotly_white'

        fig.show()

    def plot_umap(self):
        reducer = umap.UMAP(random_state=42, n_components=2)
        umap_components = reducer.fit_transform(np.asarray(self.vectors))
        self._plot_2d(umap_components[:, 0], umap_components[:, 1], self.topic_labels)