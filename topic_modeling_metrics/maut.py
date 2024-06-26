import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from topic_modeling_metrics.utils import Logger
from topic_modeling_metrics.traditional_metrics import TraditionalMetrics
from topic_modeling_metrics.cluster_metrics import ClusterMetrics

class Maut:
    def __init__(self, docs: list | np.ndarray, topic_words: dict, topic_labels: dict,
                 embedding_doc_name_path: str, embedding_word_path: str, n_words, embedding_method: str = "keybert"):
        self.docs = docs
        self.topic_words = topic_words
        self.topic_labels = topic_labels
        self.embedding_doc_name_path = embedding_doc_name_path
        self.embedding_word_path = embedding_word_path
        self.n_words = n_words
        self.embedding_method = embedding_method
        self.metric_results = []
        self.metrics_by_topic = {}
        self.metrics_dataframe = None
        self.logger = Logger("maut").get_logger()

    def get_tradicional_metrics(self):
        for method in self.topic_words:
            tm = TraditionalMetrics(self.docs, self.topic_words[method], self.n_words)

            npmis = list(tm.get_npmi().values())
            coherence = list(tm.get_coherence().values())
            embedding_distance = list(tm.get_embedding_distance(self.embedding_word_path).values())

            self.metric_results.append({
                "method": method,
                "npmi": np.mean(npmis),
                "coherence": np.mean(coherence),
                "embedding_distance": np.mean(embedding_distance)

            })

            self.metrics_by_topic[method] = {
                "npmi": npmis,
                "coherence": coherence,
                "embedding_distance": embedding_distance
            }

            self.logger.info(f"Traditional metrics for {method} calculated.")

    def get_cluster_metrics(self):
        for method in self.topic_labels:
            cm = ClusterMetrics(self.docs, self.topic_labels[method])
            cm.build_embedding(
                embedding_name_or_path=self.embedding_doc_name_path,
                embedding_method=self.embedding_method)

            self.metric_results.append({
                "method": method,
                "silhouette": cm.silhouette(),
                "beta_cv": cm.beta_cv()*-1,
                "calinski": cm.calinski()
            })

            self.logger.info(f"Cluster metrics for {method} calculated.")

    def get_maut(self):
        if not self.metric_results:
            self.get_tradicional_metrics()
            self.get_cluster_metrics()

        df = pd.DataFrame.from_dict(self.metric_results)
        df = (
            df.merge(df, on="method")[[
                "method", "npmi_x", "coherence_x", "embedding_distance_x", "silhouette_y", "calinski_y",
                "beta_cv_y"]]
              .dropna().set_index("method"))
        df.columns = [re.sub("_x|_y", "", column) for column in df.columns]
        self.metrics_dataframe = df.copy()

        scaled_values = df.values
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(scaled_values)
        scaled_values = scaled_values.T
        weights = [1 / df.shape[1]] * df.shape[1]

        for index, row in enumerate(scaled_values):
            scaled_values[index] = row * weights[index]

        scaled_values = scaled_values.T
        df_maut = pd.DataFrame([scaled_values.sum(axis=1).tolist()], columns=df.index.values)
        return df_maut

