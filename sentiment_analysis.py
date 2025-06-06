import nltk
import random
import string
from nltk.corpus import movie_reviews
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# Download NLTK data (only first time)
nltk.download('movie_reviews')

# Load movie review data
data = [(" ".join(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

# Shuffle data randomly
random.shuffle(data)

# Clean text (remove punctuation)
texts = [''.join(c for c in text if c not in string.punctuation) for text, label in data]
labels = [label for text, label in data]

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=0)

# Build a pipeline: text vectorizer + Naive Bayes classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on test data
print("=== Model Evaluation ===")
print(classification_report(y_test, model.predict(X_test)))

# Interactive loop to predict user input sentiment
print("\n=== Sentiment Prediction ===")
while True:
    user_input = input("Type a sentence (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    prediction = model.predict([user_input])[0]
    print(f"Predicted Sentiment: {prediction.capitalize()}\n")
