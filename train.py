import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from preprocess import clean_text


# Load dataset
df = pd.read_csv(
    "data/training.csv",
    encoding='latin-1',
    header=None
)


# Add column names
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Keep only needed columns
df = df[['sentiment', 'text']]

# Use smaller sample for faster training
df = df.sample(50000)

# Convert text column to string
df['text'] = df['text'].astype(str)

# Clean text
df['cleaned_text'] = df['text'].apply(clean_text)
df['sentiment'] = df['sentiment'].replace(4, 1)

# Features and labels
X = df['cleaned_text']

y = df['sentiment']


# Convert text into vectors
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = LogisticRegression()

model.fit(X_train, y_train)




import joblib

joblib.dump(model, "models/sentiment_model.pkl")

joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model and vectorizer saved") 