# News-Authenticity-Detection
# https://drive.google.com/drive/folders/1s6zo-vJPYypFHJ2amViCJ4ejprl2VKeY?usp=sharing
Fake News Detection Using Natural Language Processing and Machine Learning

Project Overview:
This project aims to classify news articles as either fake or real using natural language processing (NLP) techniques and machine learning algorithms. The approach involves preprocessing text data, creating word embeddings, and applying unsupervised clustering.

Tools and Libraries Used:
1. Python 3.x
2. Data manipulation: pandas, numpy
3. NLP: gensim, re (regex)
4. Machine Learning: scikit-learn
5. Data Visualization: matplotlib, seaborn
6. File handling: Google Colab (for mounting Google Drive)

Dataset:
The project uses two datasets: "Fake.csv" and "True.csv", containing fake and real news articles respectively. The combined dataset consists of 44,889 articles after preprocessing.

Methodology:

1. Data Preprocessing:
   - Merged title and text fields
   - Added binary labels (0 for fake, 1 for real)
   - Applied text cleaning techniques:
     * Lowercasing
     * Removing URLs, tags, punctuation, numbers, and stopwords
     * Stripping multiple whitespaces and short words

2. Word Embeddings:
   - Utilized Word2Vec model from gensim
   - Created 100-dimensional word vectors
   - Generated sentence vectors by averaging word vectors

3. Clustering:
   - Applied K-means clustering (n_clusters=2) on sentence vectors
   - Used scikit-learn's KMeans implementation

4. Dimensionality Reduction:
   - Applied Principal Component Analysis (PCA) for visualization purposes

Results and Statistics:

1. Dataset size: 44,889 articles
2. Word embedding dimension: 100
3. Number of clusters: 2
4. Clustering accuracy: 52.29%
   (Percentage of correctly clustered news articles compared to true labels)

5. Word similarity example:
   Top 5 words similar to "country":
   - nation (similarity: 0.821)
   - america (similarity: 0.667)
   - countries (similarity: 0.592)
   - europe (similarity: 0.554)
   - planet (similarity: 0.519)

6. Test case:
   A BBC news article about NASA's Mars mission was classified as fake news by the model.

Analysis:

The project demonstrates proficiency in several key areas of data science and machine learning:

1. Data preprocessing and cleaning using advanced NLP techniques
2. Implementation of word embeddings using Word2Vec
3. Application of unsupervised learning (K-means clustering)
4. Dimensionality reduction techniques (PCA) for data visualization
5. Integration of multiple Python libraries for a complete ML pipeline

The clustering accuracy of 52.29% suggests that while the model has learned some patterns, there is significant room for improvement. This could be addressed by:

1. Fine-tuning the Word2Vec model
2. Exploring other clustering algorithms or supervised learning approaches
3. Incorporating additional features beyond just text content

The misclassification of the BBC article highlights the challenges in fake news detection and the need for more sophisticated models.

Conclusion:

This project showcases a comprehensive approach to text classification using NLP and machine learning techniques. While the current accuracy is limited, the implementation demonstrates strong technical skills in data preprocessing, feature engineering, and machine learning model application. Future work could focus on improving model performance and exploring more advanced algorithms for fake news detection.
