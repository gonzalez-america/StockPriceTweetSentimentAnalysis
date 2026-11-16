#ATTEMPT 1: LOOKING AT THE CLOSING PRICE AND THE TWEETS MADE THAT DAY

import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Convert 'Date' to datetime format with timezone awareness
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
tweets_df["Date"] = tweets_df["Date"].dt.tz_convert("US/Eastern")

# Remove duplicates and missing values
tweets_df = tweets_df.drop_duplicates().dropna()

# Convert text to lowercase (optional)
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower()

# Remove mentions
tweets_df["Tweet"] = tweets_df["Tweet"].apply(lambda x: re.sub(r"@\w+", "", x))

# Download VADER lexicon
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
tweets_df["Sentiment Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Convert stock data 'Date' to datetime
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
stock_df = stock_df[["Date", "Stock Name", "Close"]]  # Keep only relevant columns

# Align tweets with the **previous trading day's closing price**
tweets_df["Trading Date"] = tweets_df["Date"].dt.date - pd.Timedelta(days=1)
stock_df["Trading Date"] = stock_df["Date"].dt.date

# Merge tweets with stock prices on Stock Name and Trading Date
merged_df = tweets_df.merge(stock_df, on=["Stock Name", "Trading Date"], how="left")

# Label if sentiment matches the stock's movement (↑ or ↓ from the previous day)
merged_df["Sentiment Label"] = merged_df["Sentiment Score"].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")

# Create a new column tracking stock movement (↑ or ↓)
merged_df["Stock Movement"] = merged_df["Close"].diff().apply(lambda x: "Up" if x > 0 else "Down" if x < 0 else "No Change")

# Check if sentiment aligns with stock movement
merged_df["Match"] = merged_df.apply(lambda row: "Yes" if (row["Sentiment Label"] == "Positive" and row["Stock Movement"] == "Up") or 
                                     (row["Sentiment Label"] == "Negative" and row["Stock Movement"] == "Down") else "No", axis=1)

# Display final dataset
print(merged_df[["Tweet", "Sentiment Score", "Sentiment Label", "Close", "Stock Movement", "Match"]].head(70))

merged_df.to_csv("/Users/alejandragonzalez/Desktop/merged_sentiment_stock_data.csv", index=False)

# Save to CSV
merged_df.to_csv("merged_sentiment_stock_data.csv", index=False)
    

#ATTEMPT 2: LOOKING AT THE CHANGE BETWEEN CLOSING TO CLOSING
#EXAMPLE:
# Tweet Date: 09-29-2022
# Change Between: 09-28-2022 and 09-29-2022

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Convert 'Date' to datetime format
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
tweets_df["Date"] = tweets_df["Date"].dt.tz_convert("US/Eastern")
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Remove duplicates and missing values
tweets_df = tweets_df.drop_duplicates().dropna()

# Convert text to lowercase (optional)
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower()

# Remove mentions
tweets_df["Tweet"] = tweets_df["Tweet"].apply(lambda x: re.sub(r"@\w+", "", x))

# Download VADER lexicon
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
tweets_df["Sentiment Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Keep relevant stock columns
stock_df = stock_df[["Date", "Stock Name", "Close"]]

# **Shift stock prices to calculate previous day's close-to-close change**
stock_df["Prev Close"] = stock_df.groupby("Stock Name")["Close"].shift(1)
stock_df["Price Change"] = stock_df["Close"] - stock_df["Prev Close"]

# **Match tweets with the previous day's price change**
tweets_df["Trading Date"] = tweets_df["Date"].dt.date  # Use tweet date as reference
stock_df["Trading Date"] = stock_df["Date"].dt.date  # Keep only date part

# Merge tweets with stock prices (tweets made **on the next day** after price change)
merged_df = tweets_df.merge(stock_df, on=["Stock Name", "Trading Date"], how="left")

# Label sentiment
merged_df["Sentiment Label"] = merged_df["Sentiment Score"].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")

# Label stock movement based on previous day's closing price change
merged_df["Stock Movement"] = merged_df["Price Change"].apply(lambda x: "Up" if x > 0 else "Down" if x < 0 else "No Change")

# Check if sentiment aligns with stock movement
merged_df["Match"] = merged_df.apply(lambda row: "Yes" if (row["Sentiment Label"] == "Positive" and row["Stock Movement"] == "Up") or 
                                     (row["Sentiment Label"] == "Negative" and row["Stock Movement"] == "Down") else "No", axis=1)

# Drop rows with missing values (if any)
merged_df = merged_df.dropna()

# Display final dataset
print(merged_df.head())

merged_df.to_csv("/Users/alejandragonzalez/Desktop/merged_sentiment_stock_data2.csv", index=False)

match_counts = merged_df["Match"].value_counts()

# Print the counts
print(match_counts)

# Compute correlation between Sentiment Score and Price Change
correlation = merged_df["Sentiment Score"].corr(merged_df["Price Change"])
print(f"Correlation between Sentiment Score and Price Change: {correlation:.4f}")


# Convert categorical target variable ('Stock Movement') to numerical
merged_df["Stock Movement"] = merged_df["Stock Movement"].map({"Up": 1, "Down": 0})

# Define features (X) and target variable (y)
X = merged_df[["Sentiment Score"]]  # Only using sentiment for now
y = merged_df["Stock Movement"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.isnull().sum())  # Check for NaN values in features
print(y_train.isnull().sum())  # Check for NaN values in target

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# ATTEMPT 3: NARROWING DOWN NUM OF COMPANIES TO THE COMPANIES THAT ARE TWEETED ABOUT THE MOST
# SAME THING AS ATTEMPT 2

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Convert 'Date' to datetime format
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
tweets_df["Date"] = tweets_df["Date"].dt.tz_convert("US/Eastern")
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Remove duplicates and missing values
tweets_df = tweets_df.drop_duplicates().dropna()

# Convert text to lowercase (optional)
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower()

# Remove mentions
tweets_df["Tweet"] = tweets_df["Tweet"].apply(lambda x: re.sub(r"@\w+", "", x))

# Download VADER lexicon
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
tweets_df["Sentiment Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Keep relevant stock columns
stock_df = stock_df[["Date", "Stock Name", "Close"]]

# **Shift stock prices to calculate previous day's close-to-close change**
stock_df["Prev Close"] = stock_df.groupby("Stock Name")["Close"].shift(1)
stock_df["Price Change"] = stock_df["Close"] - stock_df["Prev Close"]

# **Match tweets with the previous day's price change**
tweets_df["Trading Date"] = tweets_df["Date"].dt.date  # Use tweet date as reference
stock_df["Trading Date"] = stock_df["Date"].dt.date  # Keep only date part

# Merge tweets with stock prices (tweets made **on the next day** after price change)
merged_df = tweets_df.merge(stock_df, on=["Stock Name", "Trading Date"], how="left")

# Label sentiment
merged_df["Sentiment Label"] = merged_df["Sentiment Score"].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")

# Label stock movement based on previous day's closing price change
merged_df["Stock Movement"] = merged_df["Price Change"].apply(lambda x: "Up" if x > 0 else "Down" if x < 0 else "No Change")

# Check if sentiment aligns with stock movement
merged_df["Match"] = merged_df.apply(lambda row: "Yes" if (row["Sentiment Label"] == "Positive" and row["Stock Movement"] == "Up") or 
                                     (row["Sentiment Label"] == "Negative" and row["Stock Movement"] == "Down") else "No", axis=1)

# Drop rows with missing values (if any)
merged_df = merged_df.dropna()

# Display final dataset
print(merged_df.head())

merged_df.to_csv("/Users/alejandragonzalez/Desktop/merged_sentiment_stock_data4.csv", index=False)

#How many yes / nos
match_counts = merged_df["Match"].value_counts()
print(match_counts)

# Count the number of tweets per stock
tweet_counts = merged_df["Stock Name"].value_counts()
print(tweet_counts)

# Define a threshold (e.g., keep only stocks with 500+ tweets)
popular_stocks = tweet_counts[tweet_counts > 500].index
print(popular_stocks)

# Filter dataset to include only those stocks
filtered_df = merged_df[merged_df["Stock Name"].isin(popular_stocks)]

# Check the filtered dataset
print(filtered_df["Stock Name"].value_counts())

match_counts = filtered_df["Match"].value_counts()
print(match_counts)

filtered_df.to_csv("/Users/alejandragonzalez/Desktop/filtered_sentiment_stock_data3.csv", index = False)

filtered_df["Stock Movement"] = filtered_df["Stock Movement"].map({"Up": 1, "Down": 0})

correlation = filtered_df["Sentiment Score"].corr(filtered_df["Price Change"])
print(f"Correlation between Sentiment Score and Price Change: {correlation:.4f}")

correlation = filtered_df["Sentiment Score"].corr(filtered_df["Stock Movement"])
print(f"Correlation between Sentiment Score and Stock Movemennt: {correlation:.4f}")

#ATTEMPT 4: LOOKING AT THE CHANGE IN STOCK PRICE BETWEEN 1 DAYS AND THE TWEETS MADE THE FOLLIWNG DAY
#EXAMPLE:
# Tweet Date: 09-20-2022
# Change Between: 09-28-2022 and 09-29-2022

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Convert 'Date' to datetime format
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
tweets_df["Date"] = tweets_df["Date"].dt.tz_convert("US/Eastern")
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Remove duplicates and missing values
tweets_df = tweets_df.drop_duplicates().dropna()

# Convert text to lowercase
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower()

# Remove mentions
tweets_df["Tweet"] = tweets_df["Tweet"].apply(lambda x: re.sub(r"@\w+", "", x))

# Download VADER lexicon
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
tweets_df["Sentiment Score"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Keep relevant stock columns
stock_df = stock_df[["Date", "Stock Name", "Open", "Close"]]

# **Calculate same-day open-to-close price change**
stock_df["Price Change"] = stock_df["Close"] - stock_df["Open"]

# Extract only the date part for merging
tweets_df["Trading Date"] = tweets_df["Date"].dt.date  # Tweets made on a certain date
stock_df["Trading Date"] = stock_df["Date"].dt.date  # Stock prices for a specific date

# **Match tweets with the previous day's stock price change**
tweets_df["Prev Trading Date"] = tweets_df["Trading Date"] - pd.Timedelta(days=1)

# Merge tweets with stock prices based on previous trading date
merged_df = tweets_df.merge(stock_df, left_on=["Stock Name", "Prev Trading Date"], right_on=["Stock Name", "Trading Date"], how="left")

# Label sentiment
merged_df["Sentiment Label"] = merged_df["Sentiment Score"].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")

# Label stock movement based on the same-day open-to-close price change
merged_df["Stock Movement"] = merged_df["Price Change"].apply(lambda x: "Up" if x > 0 else "Down" if x < 0 else "No Change")

# Check if sentiment aligns with stock movement
merged_df["Match"] = merged_df.apply(lambda row: "Yes" if (row["Sentiment Label"] == "Positive" and row["Stock Movement"] == "Up") or 
                                     (row["Sentiment Label"] == "Negative" and row["Stock Movement"] == "Down") else "No", axis=1)

# Drop rows with missing values (if any)
merged_df = merged_df.dropna()

# Display final dataset
print(merged_df.head())

#How many yes / nos
match_counts = merged_df["Match"].value_counts()
print(match_counts)

# Count the number of tweets per stock
tweet_counts = merged_df["Stock Name"].value_counts()
print(tweet_counts)

# Define a threshold (e.g., keep only stocks with 500+ tweets)
popular_stocks = tweet_counts[tweet_counts > 500].index
print(popular_stocks)

# Filter dataset to include only those stocks
filtered_df = merged_df[merged_df["Stock Name"].isin(popular_stocks)]

# Check the filtered dataset
print(filtered_df["Stock Name"].value_counts())

match_counts = filtered_df["Match"].value_counts()
print(match_counts)

filtered_df.to_csv("/Users/alejandragonzalez/Desktop/filtered_sentiment_stock_data4.csv", index = False)

filtered_df["Stock Movement"] = filtered_df["Stock Movement"].map({"Up": 1, "Down": 0})

correlation = filtered_df["Sentiment Score"].corr(filtered_df["Price Change"])
print(f"Correlation between Sentiment Score and Price Change: {correlation:.4f}")

correlation = filtered_df["Sentiment Score"].corr(filtered_df["Stock Movement"])
print(f"Correlation between Sentiment Score and Stock Movemennt: {correlation:.4f}")


#ATTEMPT5


# Download VADER sentiment lexicon
nltk.download("vader_lexicon")

# Load your datasets
tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Convert Date columns
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Clean and preprocess tweets
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"http\S+|[^a-zA-Z\s]", "", x))

# Apply sentiment analysis
sia = SentimentIntensityAnalyzer()
tweets_df["Sentiment"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])

# Create date key for joining
tweets_df["Tweet_Date"] = tweets_df["Date"].dt.date
stock_df["Stock_Date"] = stock_df["Date"].dt.date

# Merge tweet with stock data on Stock Name and Tweet_Date
merged_df = pd.merge(
    tweets_df,
    stock_df,
    left_on=["Stock Name", "Tweet_Date"],
    right_on=["Stock Name", "Stock_Date"]
)

# Calculate price difference
merged_df["Price_Change"] = merged_df["Close"] - merged_df["Open"]
merged_df["Price_Change_Direction"] = merged_df["Price_Change"].apply(lambda x: 1 if x > 0 else 0)

# Correlation
correlation = merged_df["Sentiment"].corr(merged_df["Price_Change"])
print(f"Correlation between sentiment and price change: {correlation}")

#Bucket Analysis



