
# Sentiment Analysis: Understanding and Challenges

## Authors
- **Pavan Reddy Boyapally**  
- **Jagadeesh Chandra Bose Mende**  
- **Kavitha Madiraju**

## Guided by:
**Khaled Sayed, Assistant Professor**  
Department of Computer Science, University of New Haven  

---

## Abstract

This project aims to analyze sentiments in movie reviews, classifying them as positive, negative, or neutral using **Natural Language Processing (NLP)** techniques. We implemented and compared models like Logistic Regression, SVM, LSTM, and BERT to determine the best approach. The study addresses challenges such as sarcasm, ambiguous language, and class imbalances, providing valuable insights into audience opinions for real-world applications.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Applications of Sentiment Analysis](#applications-of-sentiment-analysis)
3. [Challenges in Sentiment Analysis](#challenges-in-sentiment-analysis)
4. [Project Objectives](#project-objectives)
5. [Datasets](#datasets)
6. [Workflow](#workflow)
7. [Data Preprocessing](#data-preprocessing)
8. [Feature Extraction](#feature-extraction)
9. [Model Selection](#model-selection)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Results](#results)
12. [Visualization](#visualization)
13. [Conclusion](#conclusion)
14. [Future Enhancements](#future-enhancements)
15. [References](#references)

---

## Introduction

**Sentiment Analysis** (or opinion mining) focuses on extracting sentiments—positive, negative, or neutral—from text data. With the rise of online platforms, sentiment analysis has become an essential tool for understanding opinions at scale.

---

## Applications of Sentiment Analysis

1. **Business and Marketing**: Understand customer feedback and improve services.
2. **Entertainment**: Predict success based on public reactions to movies and music.
3. **Politics**: Gauge public opinion on policies or figures.
4. **Healthcare**: Enhance patient care through sentiment analysis of reviews.

---

## Challenges in Sentiment Analysis

1. **Language Variability**: Informal language, slang, emojis, and abbreviations.
2. **Sarcasm and Ambiguity**: Statements like "Great! Another boring sequel" are challenging.
3. **Context Dependence**: Words like "cool" depend on context.
4. **Class Imbalance**: Uneven distribution of sentiment classes.
5. **Multilingual Texts**: Mixed-language content complicates analysis.

---

## Project Objectives

1. Build and compare machine learning and deep learning models.
2. Develop robust preprocessing techniques for noisy data.
3. Evaluate models on accuracy, precision, recall, and efficiency.
4. Propose real-world deployment strategies.

---

## Datasets

### 1. IMDB Movie Reviews
- **Description**: Contains 50,000 labeled reviews.
- **Challenges**: Spoilers, repetitive phrases.
- **Applications**: Ideal for benchmarking models.

### 2. Sentiment140 (Twitter Dataset)
- **Description**: 1.6 million labeled tweets.
- **Challenges**: Sarcasm, lack of context.
- **Applications**: Suitable for real-time social media analysis.

### 3. Amazon Product Reviews
- **Description**: 100,000 reviews with mapped star ratings.
- **Challenges**: Class imbalance, varied patterns across product categories.
- **Applications**: Understand customer opinions on e-commerce platforms.

---

## Workflow

1. **Data Collection and Preparation**:
   - Collected data from IMDB, Sentiment140, and Amazon reviews.
   - Addressed class imbalances and standardized formats.

2. **Data Preprocessing**:
   - Cleaned text, removed stopwords, and handled emojis/slang.

3. **Feature Extraction**:
   - Used TF-IDF for ML models; embeddings like Word2Vec for DL models.

4. **Model Training and Evaluation**:
   - Fine-tuned Logistic Regression, SVM, LSTM, and BERT.

5. **Visualization and Analysis**:
   - Used metrics like F1-score, confusion matrices, and precision-recall curves.

---

## Data Preprocessing

### Steps
1. Text Cleaning: Removed URLs, special characters, and converted text to lowercase.
2. Tokenization: Split sentences into individual words.
3. Stopword Removal: Removed common words like "the" and "is."
4. Lemmatization: Reduced words to their base forms (e.g., "running" → "run").
5. Handling Emojis and Slang: Replaced emojis with descriptions and expanded slang.

---

## Feature Extraction

- **TF-IDF**: Captures word importance in ML models.
- **Word Embeddings**: Used GloVe and Word2Vec for context-rich representations.
- **Transformer Tokenization**: For BERT, subword tokenization was employed.

---

## Model Selection

### Models Implemented
1. **Logistic Regression**: Simple and interpretable.
2. **SVM**: Handles high-dimensional data effectively.
3. **Naive Bayes**: Fast but assumes feature independence.
4. **LSTM**: Captures sequential dependencies in text.
5. **BERT**: Excels in understanding contextual relationships.

---

## Evaluation Metrics

1. **Accuracy**: Percentage of correct predictions.
2. **Precision**: Reliability of positive predictions.
3. **Recall**: Model's ability to identify actual positives.
4. **F1-Score**: Balance between precision and recall.
5. **Confusion Matrix**: Detailed performance breakdown.

---

## Results

| Model                   | Accuracy | Precision | Recall | F1-Score |
|-------------------------|----------|-----------|--------|----------|
| Logistic Regression     | 87.42%  | 0.88      | 0.89   | 0.88     |
| SVM                     | **91.69%** | **0.94** | **0.96** | **0.95** |
| Naive Bayes             | 84.78%  | 0.85      | 1.00   | 0.92     |
| LSTM                    | 88.78%  | 0.90      | 0.90   | 0.90     |
| BERT                    | 81.25%  | 0.84      | 0.83   | 0.83     |

### Key Observations
- **Best Performer**: SVM excelled with the highest accuracy and F1-score.
- **Most Context-Aware**: BERT was ideal for deeper context analysis but required more training data.
- **Balanced Option**: LSTM provided strong performance while being less resource-intensive.

---

## Visualization

### Precision-Recall Curves
- **SVM**: Maintained high precision and recall.
- **Naive Bayes**: Struggled with negative sentiments.
- **LSTM and BERT**: Demonstrated smoother, balanced curves.

---

## Conclusion

- **Best Model**: SVM for overall performance.
- **Most Advanced**: BERT for contextual understanding.
- **Recommended Approach**: LSTM for efficiency and reliability.

---

## Future Enhancements

1. Fine-tune BERT on larger datasets.
2. Implement real-time sentiment tracking.
3. Explore multilingual sentiment analysis.



## References

1. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."
2. Maas, A. L., et al. "Learning Word Vectors for Sentiment Analysis."
3. Mikolov, T., et al. "Efficient Estimation of Word Representations in Vector Space."
