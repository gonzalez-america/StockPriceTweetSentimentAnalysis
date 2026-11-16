
# NOTE: ATTEPTS 1-4 ARE ALL USING EXTREMEE GRADIENT BOOTING ENSEMBLE LEARNING METHOD. 

# ATTEMPT 1: 
# XGBOOST ACCURACY: .9999
#OVERFITTING WILL NOT APPLY WELL TO NEW DATA
#Precision & Recall: 1.00 for both cases  means the model perflectly identigies both rising and fallign stock movmeents
#F 1 Score: Balace between precisiona and recall
# Support: THe dataset seems balanced between the two classes

import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Convert 'Date' to datetime format
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
tweets_df["Date"] = tweets_df["Date"].dt.tz_convert("US/Eastern")

# Remove duplicates and missing values
tweets_df = tweets_df.drop_duplicates().dropna()

# Convert text to lowercase
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower()

# Remove mentions from tweets
tweets_df["Tweet"] = tweets_df["Tweet"].apply(lambda x: re.sub(r"@\w+", "", x))

# Download VADER lexicon and apply sentiment analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
tweets_df["Sentiment Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
tweets_df["Positive Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["pos"])
tweets_df["Negative Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["neg"])

# **New Tweet-Based Features**
tweets_df["Tweet Length"] = tweets_df["Tweet"].apply(len)
tweets_df["Exclamation Count"] = tweets_df["Tweet"].apply(lambda x: x.count("!"))

# Convert stock data 'Date' to datetime and keep relevant columns
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
stock_df = stock_df[["Date", "Stock Name", "Open", "Close", "Volume"]]

# **Calculate new stock-based features**
stock_df["Prev Close"] = stock_df.groupby("Stock Name")["Close"].shift(1)  # Previous day's close
stock_df["Price Change"] = stock_df["Close"] - stock_df["Open"]  # Open-to-Close price change
stock_df["Pct Change"] = (stock_df["Price Change"] / stock_df["Open"]) * 100  # Percentage change

# Create stock movement label (Up = 1, Down = 0)
stock_df["Stock Movement"] = stock_df["Price Change"].apply(lambda x: 1 if x > 0 else 0)

# Merge tweets with stock prices using the previous day's stock data
tweets_df["Prev Trading Date"] = tweets_df["Date"].dt.date - pd.Timedelta(days=1)
stock_df["Trading Date"] = stock_df["Date"].dt.date

merged_df = tweets_df.merge(stock_df, left_on=["Stock Name", "Prev Trading Date"], right_on=["Stock Name", "Trading Date"], how="left")

# Drop NaN values after merge
merged_df = merged_df.dropna()

# **Final Feature Selection**
features = ["Sentiment Score", "Positive Score", "Negative Score", "Tweet Length", "Exclamation Count", 
            "Prev Close", "Price Change", "Pct Change", "Volume"]

X = merged_df[features]
y = merged_df["Stock Movement"]  # Target: 1 (Up), 0 (Down)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Initialize and Train Models**
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Save final merged dataset
merged_df.to_csv("enhanced_sentiment_stock_data.csv", index=False)


#ATTEMPT 2:
# Open-to-Close Price Change
#Still too Fitting with same results
# Accurac = 0.9994
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load datasets
tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Convert 'Date' to datetime format
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True).dt.tz_convert("US/Eastern")
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Remove duplicates and missing values
tweets_df.drop_duplicates(inplace=True)
tweets_df.dropna(inplace=True)

# Clean and preprocess tweets
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"@\w+", "", x))

# Sentiment Analysis using VADER
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
tweets_df["Sentiment Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Feature Engineering - Add stock movement based on Open to Close Change
stock_df["Price Change"] = stock_df["Close"] - stock_df["Open"]
stock_df["Stock Movement"] = stock_df["Price Change"].apply(lambda x: 1 if x > 0 else 0)  # Up (1) / Down (0)

# Align tweets with the **next trading day's stock movement**
tweets_df["Trading Date"] = tweets_df["Date"].dt.date
stock_df["Trading Date"] = stock_df["Date"].dt.date

# Merge tweets with stock prices
merged_df = tweets_df.merge(stock_df, on=["Stock Name", "Trading Date"], how="left")

# Drop rows with missing values (if any)
merged_df.dropna(inplace=True)

# Select Features
features = ["Sentiment Score", "Price Change"]
X = merged_df[features]
y = merged_df["Stock Movement"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **Hyperparameter tuning for XGBoost**
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
param_dist = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.1, 0.3],
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=3, scoring="accuracy", n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best Model
best_xgb = random_search.best_estimator_

# Model Evaluation
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Optimized XGBoost Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 3 ATTEMPT 3: ONLY USING THE MOST TWEETED STOCKS > 1000
# See if we can apply this better to stocks that are tweeted about the most
# Accuracy increased to .9997

# Load datasets
tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Convert 'Date' to datetime format
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True).dt.tz_convert("US/Eastern")
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Remove duplicates and missing values
tweets_df.drop_duplicates(inplace=True)
tweets_df.dropna(inplace=True)

# Clean and preprocess tweets
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"@\w+", "", x))

# Sentiment Analysis using VADER
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
tweets_df["Sentiment Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Identify stocks with more than 1000 tweets
popular_stocks = tweets_df["Stock Name"].value_counts()
popular_stocks = popular_stocks[popular_stocks > 1000].index.tolist()

# Filter dataset for only those stocks
tweets_df = tweets_df[tweets_df["Stock Name"].isin(popular_stocks)]
stock_df = stock_df[stock_df["Stock Name"].isin(popular_stocks)]

# Feature Engineering - Add stock movement based on Open to Close Change
stock_df["Price Change"] = stock_df["Close"] - stock_df["Open"]
stock_df["Stock Movement"] = stock_df["Price Change"].apply(lambda x: 1 if x > 0 else 0)  # Up (1) / Down (0)

# Align tweets with the **next trading day's stock movement**
tweets_df["Trading Date"] = tweets_df["Date"].dt.date
stock_df["Trading Date"] = stock_df["Date"].dt.date

# Merge tweets with stock prices
merged_df = tweets_df.merge(stock_df, on=["Stock Name", "Trading Date"], how="left")

# Drop rows with missing values (if any)
merged_df.dropna(inplace=True)

# Select Features
features = ["Sentiment Score", "Price Change"]
X = merged_df[features]
y = merged_df["Stock Movement"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **Train XGBoost with optimized parameters**
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, learning_rate=0.1, max_depth=5)
xgb.fit(X_train, y_train)

# Model Evaluation
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Optimized XGBoost Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# ATTEMPT 4: Open to Close Price Change and Tweets Made the following Day
# Keeping Tweets over 1000
# Accuracy Decreased : .9832
#Precision Decreased to : .98
#  Recall Decreased to : .98
# F-1 Score Decreased to: .98

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# This will make sure that all the dates are in Eastern Time
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True).dt.tz_convert("US/Eastern")
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Remove duplicates and missing values
tweets_df.drop_duplicates(inplace=True)
tweets_df.dropna(inplace=True)

# Makes text lower case and it removes the twitter handles
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"@\w+", "", x))

# Sentiment Analysis using VADER
# VADER has a scare of -1 to 1 negative to positive sentiment of the Tweet
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
tweets_df["Sentiment Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Identify stocks with more than 1000 tweets
#This are the "most popular stocks"
popular_stocks = tweets_df["Stock Name"].value_counts()
popular_stocks = popular_stocks[popular_stocks > 1000].index.tolist()

# Filter dataset for only those stocks
tweets_df = tweets_df[tweets_df["Stock Name"].isin(popular_stocks)]
stock_df = stock_df[stock_df["Stock Name"].isin(popular_stocks)]

# Feature Engineering - Add stock movement based on Open to Close Change
# This aligns tweets with stock data from the text trading day meaning tweets date minus one day

stock_df["Price Change"] = stock_df["Close"] - stock_df["Open"]
stock_df["Stock Movement"] = stock_df["Price Change"].apply(lambda x: 1 if x > 0 else 0)  # Up (1) / Down (0)

# Align tweets with the **NEXT trading day's stock movement**
# Can be used to predict stock movement ont eh following day
# Maybe change this so that it predicts the sentiment based on the stock price from the prior day??

tweets_df["Trading Date"] = tweets_df["Date"].dt.date
tweets_df["Stock Date"] = tweets_df["Trading Date"] - pd.Timedelta(days=1)  # Match to the previous day's stock price

stock_df["Trading Date"] = stock_df["Date"].dt.date  # Keep only the date

# Merge tweets with stock prices (tweets made **on the day after** stock closes)
merged_df = tweets_df.merge(stock_df, left_on=["Stock Name", "Stock Date"], right_on=["Stock Name", "Trading Date"], how="left")

print(merged_df.head)

# Drop rows with missing values (if any)
merged_df.dropna(inplace=True)

# Select Features
features = ["Sentiment Score", "Price Change"]
X = merged_df[features]
y = merged_df["Stock Movement"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost with optimized parameters
# This means that the classifier is trained to predict stock movement based on sentiment and stock price
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, learning_rate=0.1, max_depth=5)
xgb.fit(X_train, y_train)

# Model Evaluation
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Optimized XGBoost Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# ATTEMPT 5: Random Forest