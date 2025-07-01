# import re
# import string
# import nltk
# import pickle
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from imblearn.over_sampling import RandomOverSampler
# from textblob import TextBlob

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# df = pd.read_csv("user_courses_review_09_2023.csv",on_bad_lines='skip')
# df.drop_duplicates(inplace=True)
# df.dropna(subset=['review_comment'], inplace=True)

# df['polarity'] = df['review_comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
# df['label'] = df['polarity'].apply(lambda x: 1 if x > 0 else 0)

# class TextPreprocessor(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.stop_words = set(stopwords.words("english"))
#         self.lemmatizer = WordNetLemmatizer()

#     def clean_text(self, text):
#         text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
#         tokens = word_tokenize(text)
#         tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words and w not in string.punctuation]
#         return ' '.join(tokens)

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X.apply(self.clean_text)

# pipeline = Pipeline([
#     ('preprocess', TextPreprocessor()),
#     ('tfidf', TfidfVectorizer(max_features=5000)),
#     ('nb', MultinomialNB())
# ])

# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(df[['review_comment']], df['label'])
# pipeline.fit(X_resampled['review_comment'], y_resampled)

# with open("nb_pipeline_with_cleaning.pkl", "wb") as f:
#     pickle.dump(pipeline, f)

# print("✅ Model training complete. Pipeline saved to 'nb_pipeline_with_cleaning.pkl'.")


import re
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

data = pd.read_csv('user_courses_review_09_2023.csv', on_bad_lines='skip')
# data.head()

# Check current data types
dtypes_before = data.dtypes

# Convert 'review_rating' to numeric (force errors to NaN)
data['review_rating'] = pd.to_numeric(data['review_rating'], errors='coerce')

# Re-check data types after conversion
dtypes_after = data.dtypes

print(dtypes_before,'\n\n',dtypes_after)


# Check missing values
print("\nMissing Values:\n", data.isnull().sum())

# Check for duplicate rows
duplicates = data.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Optionally drop rows with missing 'review_rating' or 'course_name'
df_cleaned = data.dropna(subset=['review_rating', 'course_name'])


# Drop duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

# Optional: reset index
df_cleaned.reset_index(drop=True, inplace=True)

duplicates = df_cleaned.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Confirm result
print(f"Shape after removing duplicates: {df_cleaned.shape}")


df_cleaned.dropna(subset=['review_comment'], inplace=True)


# Label based on TextBlob polarity (already done previously)
from textblob import TextBlob
df_cleaned['polarity'] = df_cleaned['review_comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
df_cleaned['label'] = df_cleaned['polarity'].apply(lambda x: 1 if x > 0 else 0)



# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = re.sub(r"[^a-zA-Z]", " ", text.lower()) # Lowercase and remove non-alphabetic characters
        tokens = nltk.word_tokenize(text) # Tokenize
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and word not in string.punctuation]
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text)


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Build the full pipeline
nb_pipeline = Pipeline([
    ('preprocess', TextPreprocessor()),
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('nb', MultinomialNB())
])

# Train on cleaned_comment and label
# nb_pipeline.fit(df_cleaned['review_comment'], df_cleaned['label'])  # Use original text here

from imblearn.over_sampling import RandomOverSampler

# Oversample raw data BEFORE vectorizing
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df_cleaned[['review_comment']], df_cleaned['label'])

# Fit the pipeline on resampled text
nb_pipeline.fit(X_resampled['review_comment'], y_resampled)


import pickle

# Save pipeline
with open("nb_pipeline_with_cleaning.pkl", "wb") as f:
    pickle.dump(nb_pipeline, f)

print("✅ Model training complete. Pipeline saved to 'nb_pipeline_with_cleaning.pkl'.")