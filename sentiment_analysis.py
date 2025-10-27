import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

print("📂 Loading dataset...")
df = pd.read_csv("dataset.csv")

print("✅ Dataset loaded successfully!")
print(df.head())

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.lower().split()
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return " ".join(words)

df['cleaned_review'] = df['review'].apply(preprocess_text)

X = df['cleaned_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

while True:
    review = input("\nEnter a movie review (or 'exit' to stop): ")
    if review.lower() == 'exit':
        break
    cleaned = preprocess_text(review)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    print("Sentiment:", prediction[0])
