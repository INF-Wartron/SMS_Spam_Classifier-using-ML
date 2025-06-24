# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only necessary columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Hyperparameter tuning for Naive Bayes
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
grid = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train_tfidf, y_train)

# Best model after tuning
best_nb = grid.best_estimator_

# Predict and evaluate
y_pred = best_nb.predict(X_test_tfidf)

# Collect results in the expected format
results = {
    "Naive Bayes": {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Report": classification_report(y_test, y_pred)
    }
}

# Print results
for name, metrics in results.items():
    print(f"\n--- {name} ---")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print("Classification Report:\n", metrics["Report"])