
## Overview

The project aims to develop a  model that accurately classifies tweets as either related to real disasters or not. This task addresses the growing importance of Twitter as a real-time emergency communication channel, particularly due to the widespread use of smartphones. The model will help disaster relief organizations and news agencies efficiently monitor and filter Twitter content during emergencies. Working with a dataset of 10,000 hand-classified tweets, the challenge lies in distinguishing between genuine disaster announcements and unrelated content, thereby improving emergency response times, reducing misinformation, and enhancing public awareness during critical situations.

## Repository Contents
- `data/`: Contains the dataset used for training and testing, as well as the submission files for kaggle 
- `notebooks/`: Jupyter notebook with the project's code and analysis.
- `models/`: Saved model files
- `src/`: Source code for Gradio web application for model deployment
- `requirements.txt`: List of required Python packages

## Methodology

### Preliminary Analysis
1. Identified percentage of missing value and Examined balance of disaster vs. non-disaster tweets.

2. Performed text-level analysis by examining both word and character metrics. For word-level analysis, I calculated the word count and determined the average word length. For character-level analysis, I measured the character count both with and without spaces.
   
### Text Preprocessing
1. Cleaned text by removing URLs, HTML tags, @mentions, #hashtags, punctuations, website names and special characters.
   
2. Implemented WhitespaceTokenizer for precise word splitting.
 
3. Utilized stopwords to eliminate common words to focus on meaningful content.
 
4. Reduced words to their base form using Lemmatization for better feature extraction.

### Feature Extraction
1. Implemented Bag-of-Words (BoW) to represent text data as vectors based on word frequency.
   
2. Implemented Term Frequency - Inverse Document Frequency (TF-IDF) by assigning weights to words by their importance, considering their frequency in documents and across the dataset, using an n-gram range of (1, 2).
   
3. Trained Word2Vec specifically on our dataset and generated word embeddings to capture semantic relationships between words. Set vocabulary size to 5000 to limit computations a bit.
     
### Modeling and Evaluation
1. Employed both machine learning (ML) and deep learning (DL) models such as Logistic Regression, Naive Bayes, LSTM's(MultiLayer and Bidirectional) and GRU's(MultiLayer and Bidirectional).
2. Assessed the performance of the models on our dataset using metrics such as accuracy, precision, recall and F1-score.

## Installation and Usage
1. Clone the repository.
```bash
git clone https://github.com/Shanmukhi1920/Text_Classification
cd Text_Classification
```
2. Install required packages.
```bash
pip install -r requirements.txt
```
3. Run Jupyter notebooks to view the analysis.
```bash
jupyter notebook
```
Then, open `disaster-tweets.ipynb` in Jupyter in the `notebooks/` directory.
4. Launch the gradio web app
```bash
python src/app.py
```
## Deployment
The model is deployed as a web application using Gradio and hosted on Hugging Face Spaces. You can access the live demo [here](https://huggingface.co/spaces/Shanmukhi0109/Disaster_Tweet_Classifier)

## Key Findings
1. Logistic Regression with TF-IDF preprocessing achieved the highest F1 score of 0.79007 on the test data, closely followed by Logistic Regression with Bag-of-Words (BOW) preprocessing at 0.78853. 
2. Among the recurrent neural network models using Word2Vec embeddings, GRU achieved an F1 score of 0.77536, slightly outperforming LSTM (0.77382) and SimpleRNN (0.73214).
3. Naive Bayes with TF-IDF preprocessing achieved an F1 score of 0.77873 on the test data but demonstrated a lower false negative rate during validation, making it the chosen model for deployment.

## Citations
Addison Howard, devrishi, Phil Culliton, Yufeng Guo. (2019). Natural Language Processing with Disaster Tweets. Kaggle. https://kaggle.com/competitions/nlp-getting-started

