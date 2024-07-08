
# Overview

Twitter has become  an  important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce  an  emergency  they’re  observing  in real-time. Because of this, more agencies are interested in programmatically monitoring  Twitter (i.e. disaster relief organizations and news agencies). But it’s  not always clear whether a person’s words are actually announcing a disaster.

In this project, I am challenged to build a model that predicts which Tweets are about real disasters and which one’s aren’t. The dataset consists of 10,000 tweets that were hand classified.

# Repository Contents
- `data/`: Contains the dataset used for training and testing, as well as the submission files for kaggle 
- `notebooks/`: Jupyter notebook with the project's code and analysis.
- `models/`: Saved model files
- `src/`: Source code for Gradio web application for model deployment
- `requirements.txt`: List of required Python packages

# Methodology

## Preliminary Analysis
1. Identified percentage of missing value and Examined balance of disaster vs. non-disaster tweets.

2. Performed text-level analysis by examining both word and character metrics. For word-level analysis, I calculated the word count and determined the average word length. For character-level analysis, I measured the character count both with and without spaces.
   
## Text Preprocessing
1. Cleaned text by removing URLs, HTML tags, @mentions, #hashtags, punctuations, website names and special characters.
   
2. Implemented WhitespaceTokenizer for precise word splitting.
 
3. Utilized stopwords to eliminate common words to focus on meaningful content.
 
4. Reduced words to their base form using Lemmatization for better feature extraction.

## Feature Extraction
1. Implemented Bag-of-Words (BoW) to represent text data as vectors based on word frequency.
   
2. Implemented Term Frequency - Inverse Document Frequency (TF-IDF) by assigning weights to words by their importance, considering their frequency in documents and across the dataset, using an n-gram range of (1, 2).
   
3. Trained Word2Vec specifically on our dataset and generated word embeddings to capture semantic relationships between words. Set vocabulary size to 5000 to limit computations a bit.
     
## Modeling and Evaluation
1. Employed both machine learning (ML) and deep learning (DL) models such as Logistic Regression, Naive Bayes, LSTM's and GRU's.
2. Assessed the performance of the models on our dataset using metrics such as accuracy, precision, recall and F1-score.

# Installation and Usage
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
# Deployment
The model is deployed as a web application using Gradio and hosted on Hugging Face Spaces. You can access the live demo [here](https://huggingface.co/spaces/Shanmukhi0109/Disaster_Tweet_Classifier)

# Key Findings
1. Naive Bayes emerged as the top performer with an accuracy of 80%, precision of 79%, recall of 82%, and an F1-score of 0.80.
2. Logistic Regression demonstrated competitive performance with an accuracy of 77%, precision of 76%, recall of 79%, and an F1-score of 0.77.
3. LSTM and GRU models showed lower accuracy rates of 62.9% and 61.5%, respectively, indicating the need for more data or additional tuning.

# Future Work:
Explore further tuning of LSTM and GRU models and consider using larger datasets to leverage their potential fully.












# Citations
Addison Howard, devrishi, Phil Culliton, Yufeng Guo. (2019). Natural Language Processing with Disaster Tweets. Kaggle. https://kaggle.com/competitions/nlp-getting-started

