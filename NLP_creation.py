import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import spacy

# Load data
def get_nlp_train(train_path, test_path):
    """Loads NLP training and test datasets from CSV files."""
    train_data = pd.read_csv(train_path, encoding="utf-8", engine="python")
    test_data = pd.read_csv(test_path, encoding="utf-8", engine="python")

    # Normalize column names (strip spaces, lowercase)
    train_data.columns = train_data.columns.str.strip().str.lower()
    test_data.columns = test_data.columns.str.strip().str.lower()

    return train_data, test_data
train_df, test_df = get_nlp_train("training.csv", "testing.csv")

# Ensure the required columns are present
required_columns = ["text", "label"]
for col in required_columns:
    if col not in train_df.columns or col not in test_df.columns:
        raise ValueError(f"Missing required column: {col}")

# Check for missing labels and drop them
train_df = train_df.dropna(subset=["label"])
test_df = test_df.dropna(subset=["label"])

# Convert labels to numeric (if they are strings like "gibberish" and "normal")
if train_df["label"].dtype == object:
    train_df["label"] = train_df["label"].astype(str).str.strip().map({"gibberish": 0, "normal": 1})
if test_df["label"].dtype == object:
    test_df["label"] = test_df["label"].astype(str).str.strip().map({"gibberish": 0, "normal": 1})

# Ensure labels are numeric
train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce")
test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce")

# Load NLP model for text preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Tokenizes and cleans text using SpaCy."""
    if pd.isna(text):  # Handle missing text
        return ""
    doc = nlp(text.lower())  # Convert to lowercase
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct] #lemmaization for the cleaned/stripped setences
    return " ".join(tokens) #tokenized text
# Apply preprocessing to datasets
train_df["processed_text"] = train_df["text"].astype(str).apply(preprocess_text)
test_df["processed_text"] = test_df["text"].astype(str).apply(preprocess_text)

# Run the NLP model
def classify_words(train_df, testing_df, text_column, label_column):
    """Trains a classifier to distinguish gibberish from non-gibberish text."""
    x_train, y_train = train_df[text_column], train_df[label_column] #give data for training x = text, y=output
    x_test, y_test = testing_df[text_column], testing_df[label_column]# give data for testing

    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500))
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(predictions)
    acc = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {acc:.4f}")
    return model

# Train and print the model
nlp_model = classify_words(train_df, test_df, "processed_text", "label")
#print(nlp_model)

print(train_df.sample(5))  # Randomly check some examples
print(test_df.sample(5))
