#Load libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


# Read in dataset
file_path = r"C:\Users\nabe7\Documents\COMP_4450\COMP_4450_Assignment_1_Nathaniel_Benton\IMDB Dataset.csv"
df = pd.read_csv(file_path)

# Split into features
X = df['review']
y = df['sentiment']

# Setting TfidfVectorizer
vectorizer = TfidfVectorizer()

# Create the pipeline
imbd_transformer = \
Pipeline([('vectorize_text', vectorizer),
        ('classifier', MultinomialNB())
          ])

# Fit the model
imbd_transformer.fit(X, y)

# Save the model to a file
joblib.dump(imbd_transformer, 'sentiment_model.pkl')
print("Model trained and saved!")