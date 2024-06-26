# Topic Modeling Metrics

This project is a collection of Python scripts that can be used to evaluate the quality of topic models. The scripts are based on the following metrics:

* Traditional Metrics

    - **NPMI**: Normalized Pointwise Mutual Information is a measure of the semantic similarity between high-scoring words in a topic. The higher the NPMI, the better the topic.
    - **Coherence**: This metric measures the semantic similarity between high-scoring words in a topic. The higher the coherence, the better the topic.
    - **Embedding similarity**: This metric measures the similarity between high-scoring words in a topic using pre-trained word embeddings. The higher the similarity, the better the topic.

* Cluster Metrics

    - **Silhouette Score**: This metric measures how similar an object is to its own cluster compared to other clusters. The higher the silhouette score, the better the clustering.
    - **Calinski-Harabasz Index**: This metric measures the ratio of the sum of between-cluster dispersion and within-cluster dispersion. The higher the Calinski-Harabasz index, the better the clustering.
    - **BetaCV**: This metric measures the ratio of the sum of within-cluster dispersion and between-cluster dispersion. The lower the BetaCV, the better the clustering.


## Installation

This project requires Python and pip installed. Clone the project and install the dependencies:

```bash
    pip install git+https://github.com/antonio258/topic_modeling_metrics.git
```

## Usage

Import the module and generate Maut and metrics.

```python

from topic_modeling_metrics.maut import Maut

maut = Maut(
    docs=docs, # document list
    topic_words=topic_words, # list of list of words, each list is a topic
    topic_labels=topic_labels, # 
    embedding_word_path=embedding_word_path, # path to the word embeddings (FastText, Glove, Word2Vec)
    embedding_doc_name_path=enbedding_doc_name_path, # path or name to the document embeddings. (sentenceBert models)
    n_words=10
    ) # number of words, by topic, to consider in the metrics.

df_result = maut.get_maut()
df_metrics_results = maut.metrics_dataframe

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.