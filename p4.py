# Practical 4
# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample documents
documents = [
    {"text": "The Yankees won the game last night", "label": "sports"},
    {"text": "The president gave a speech today", "label": "politics"},
    {"text": "The Lakers are on a winning streak", "label": "sports"},
    {"text": "The government announced a new policy", "label": "politics"},
    {"text": "The Cowboys lost to the Eagles", "label": "sports"},
    {"text": "The economy is growing rapidly", "label": "politics"},
    {"text": "The Bulls are on a losing streak", "label": "sports"},
    {"text": "The president is visiting a foreign country", "label": "politics"},
    {"text": "The Packers won the Super Bowl", "label": "sports"},
    {"text": "The government is facing a crisis", "label": "politics"}
]

# Prepare training and test data
train_docs, test_docs, train_labels, test_labels = train_test_split(
    [doc["text"] for doc in documents],
    [doc["label"] for doc in documents],
    test_size=0.3,
    random_state=42
)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_docs)
X_test = vectorizer.transform(test_docs)

# Create and train the classifier
clf = MultinomialNB()
clf.fit(X_train, train_labels)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(test_labels, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(test_labels, y_pred))

# Predict new documents
new_doc = "The Red Sox won the World Series"
new_doc_vector = vectorizer.transform([new_doc])
prediction = clf.predict(new_doc_vector)
print("Prediction for new document:", prediction)

new_doc = "The president gave a speech today"
new_doc_vector = vectorizer.transform([new_doc])
prediction = clf.predict(new_doc_vector)
print("Prediction for new document:", prediction)
