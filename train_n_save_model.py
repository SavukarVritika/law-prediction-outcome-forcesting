import pickle
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv('draft_mini_dataset.csv', encoding='latin-1')

# Step 2: Prepare textual data and target variables
X = df['Case Facts']  # Input feature (textual data)
y_law = df['Relevant Law']  # Target for Relevant Law
y_outcome = df['Outcome']  # Target for Outcome

# Step 3: Split the data into training and testing sets
X_train, X_test, y_law_train, y_law_test = train_test_split(X, y_law, test_size=0.2, random_state=42)
_, _, y_outcome_train, y_outcome_test = train_test_split(X, y_outcome, test_size=0.2, random_state=42)

# ***Step 4: Handle missing values in the target variables (y_law_train, y_outcome_train)***
# Remove rows with missing values in the target variables
X_train = X_train[y_law_train.notna()]  # Filter X_train based on non-missing y_law_train
y_law_train = y_law_train.dropna()  # Drop missing values from y_law_train
X_train = X_train[y_outcome_train.notna()]  # Filter X_train based on non-missing y_outcome_train
y_outcome_train = y_outcome_train.dropna()  # Drop missing values from y_outcome_train

# Ensure X_test and y_test are aligned after dropping NaNs in training data
X_test = X_test[y_law_test.notna()]
y_law_test = y_law_test.dropna()
X_test = X_test[y_outcome_test.notna()]
y_outcome_test = y_outcome_test.dropna()

# Step 4: Convert text data to numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_tfidf = tfidf_vectorizer.transform(X_test)  # Transform testing data

# Step 5: Train models using Gradient Boosting Classifier
model_law = GradientBoostingClassifier(random_state=42)
model_law.fit(X_train_tfidf, y_law_train)

model_outcome = GradientBoostingClassifier(random_state=42)
model_outcome.fit(X_train_tfidf, y_outcome_train)



# Save the models
with open('model_law.pkl', 'wb') as f:
    pickle.dump(model_law, f)

with open('model_outcome.pkl', 'wb') as f:
    pickle.dump(model_outcome, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)




# Step 6: Evaluate models
y_law_pred = model_law.predict(X_test_tfidf)
law_accuracy = accuracy_score(y_law_test, y_law_pred)
print(f"Relevant Law Model Accuracy: {law_accuracy * 100:.2f}%")

y_outcome_pred = model_outcome.predict(X_test_tfidf)
outcome_accuracy = accuracy_score(y_outcome_test, y_outcome_pred)
print(f"Outcome Model Accuracy: {outcome_accuracy * 100:.2f}%")

# Step 7: Function to check if the input is valid (exists in the dataset)
def is_input_valid(new_input, X_train):
    """Check if the new input is similar to any of the training data."""
    # Transform new input into TF-IDF representation
    new_input_tfidf = tfidf_vectorizer.transform([new_input])  # Transform the new input

    # Calculate similarity between the new input and the training data
    similarity_scores = cosine_similarity(new_input_tfidf, tfidf_vectorizer.transform(X_train))

    # Get maximum similarity score
    max_similarity = np.max(similarity_scores)

    # Define threshold for similarity (e.g., 0.5)
    return max_similarity >= 0.5

# Step 8: Dynamic input from user
new_input = input("Enter the case facts for prediction: ")  # Dynamic input from user

# Check if the input exists in the training dataset
if is_input_valid(new_input, X_train):
    # If input is valid, transform and predict
    new_input_tfidf = tfidf_vectorizer.transform([new_input])  # Transform input to TF-IDF format

    predicted_law = model_law.predict(new_input_tfidf)[0]
    predicted_outcome = model_outcome.predict(new_input_tfidf)[0]

    print(f"Predicted Relevant Law: {predicted_law}")
    print(f"Predicted Outcome: {predicted_outcome}")
else:
    # If input is out of bounds
    print("Error: Input is out of bounds or too different from training data.")


