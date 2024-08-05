import joblib
import re
import gradio as gr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load your model
vectorizer = joblib.load('D:/Text_Classification/models/vectorizer.pkl')
model = joblib.load('D:/Text_Classification/models/naive_bayes.pkl')

def clean_text(text):
    # convert text to lowercase
    text = text.lower()
    # remove url's
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Handle website names (e.g., HorrorMovies.ca)
    text = re.sub(r'\b[A-Za-z0-9-]+\.[a-z]{2,}\b', '', text)
    # remove html tags
    text = re.sub('<.*?>+', '', text)
    # Handle hashtags: remove the # symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    # Handle mentions: remove @ symbol and username
    text = re.sub(r'@\w+', '', text)
    # Remove emojis and special characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove standalone numbers, but keep numbers within words
    text = re.sub(r'\b\d+\b', '', text)
    # Remove square brackets
    text = text.replace('[', '').replace(']', '')
    # Remove punctuation, but keep apostrophes within words and hyphens between words
    text = re.sub(r"[^\w\s'-]|(?<!\w)['|-]|['|-](?!\w)", '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# preprocessing function 
def preprocess_text(text):
    # Tokenization
    wt = WhitespaceTokenizer()
    text = wt.tokenize(text)
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    # Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word,pos='v') for word in text]
    # TF-IDF Vectorization
    text = vectorizer.transform(text)
    return text

# prediction function
def predict(text):
    cleaned_text = clean_text(text)
    processed_text = preprocess_text(cleaned_text)
    prediction = model.predict(processed_text)[0]
    if prediction == 1:
        return "Disaster-related"
    else: 
        return "Not disaster-related"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter tweet here..."),
    outputs="text",
    title="Disaster Tweet Classifier",
    description="Enter a tweet to classify if it's disaster-related or not."
)

# Launch the app
iface.launch()
