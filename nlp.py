import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the cleaned dataset from Member 1
df = pd.read_csv('review_cleaned_output.csv')

# Step 1: Tokenize the review_cleaned column
df['tokens'] = df['review_cleaned'].astype(str).apply(lambda x: word_tokenize(x, preserve_line=True))

# Step 2: Remove stopwords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Step 3: Lemmatize words
lemmatizer = WordNetLemmatizer()
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Step 4: Rejoin tokens into a single processed review
df['processed_review'] = df['tokens'].apply(lambda x: ' '.join(x))

# Step 5: Save final output
df.to_csv('review_nlp_ready.csv', index=False)
print("NLP preprocessing complete. Output saved as: review_nlp_ready.csv")