# Import necessary libraries
import pandas as pd

# Load the NLP-processed dataset
df = pd.read_csv('review_nlp_ready.csv')

# Separate the input feature (processed review text) and target labels (sentiment)
X_text = df['processed_review']
y_labels = df['Sentiment']

# TF-IDF Vectorization to convert text data into numeric features
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 most relevant terms
X_tfidf = vectorizer.fit_transform(X_text)       # Transform the text into TF-IDF vectors

# Encode the target labels (Sentiment: Positive, Neutral, Negative) into numeric values
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_labels)

# Split the dataset into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42
)

# Save the processed data for the Model Evaluator (Member 4)
import joblib
from scipy.sparse import save_npz

save_npz("X_train.npz", X_train)   # Save training features
save_npz("X_test.npz", X_test)     # Save test features
joblib.dump(y_train, "y_train.pkl") # Save training labels
joblib.dump(y_test, "y_test.pkl")   # Save test labels
