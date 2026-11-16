
import pandas as pd
from IPython.display import display
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from textblob import TextBlob
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from sklearn.metrics import RocCurveDisplay

nltk.download("vader_lexicon")


tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# CLEAN DATA

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

# Shift stock data so tweet aligns with next trading day's price
# stock_df["Stock_Date"] = stock_df["Stock_Date"] - pd.Timedelta(days=1)

# Merge tweet with stock data on Stock Name and Tweet_Date
merged_df = pd.merge(
    tweets_df,
    stock_df,
    left_on=["Stock Name", "Tweet_Date"],
    right_on=["Stock Name", "Stock_Date"]
)

# Calculate price change and direction
merged_df["Price_Change"] = merged_df["Close"] - merged_df["Open"]
merged_df["Price_Change_Direction"] = merged_df["Price_Change"].apply(lambda x: 1 if x > 0 else 0)

#********************************************************************
# CORRELATION ANALYSIS

# Correlation
correlation = merged_df["Sentiment"].corr(merged_df["Price_Change"])
print(f"Correlation between tweet sentiment and next day's price change: {correlation}")

#********************************************************************
#CORRELATION ANALYSIS BY STOCK

correlations = merged_df.groupby("Stock Name").apply(
    lambda group: group["Sentiment"].corr(group["Price_Change"])
).sort_values(ascending=False)

print(correlations)

# CORRELATION VISUALIZATION

# Set plot size and style
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# Reset index for plotting
correlations_df = correlations.reset_index()
correlations_df.columns = ["Stock Name", "Correlation"]

# Create horizontal bar plot
sns.barplot(
    x="Correlation",
    y="Stock Name",
    data=correlations_df,
    palette="coolwarm",
    edgecolor=".3"
)

# Add labels and title
plt.title("Correlation Between Tweet Sentiment and Price Change by Stock", fontsize=14)
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Stock Name")

# Show plot
plt.tight_layout()
plt.show()

#BUCKET ANALYSIS 

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Preprocessing
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"http\S+|[^a-zA-Z\s]", "", x))

# Sentiment
sia = SentimentIntensityAnalyzer()
tweets_df["Sentiment"] = tweets_df["Tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])

# Create Key

tweets_df["Tweet_Date"] = tweets_df["Date"].dt.date
stock_df["Stock_Date"] = stock_df["Date"].dt.date

# Shift stock data so tweet aligns with next trading day's price
stock_df["Stock_Date"] = stock_df["Stock_Date"] - pd.Timedelta(days=1)

# Merge
merged_df = pd.merge(
    tweets_df,
    stock_df,
    left_on=["Stock Name", "Tweet_Date"],
    right_on=["Stock Name", "Stock_Date"]
)

merged_df["Price_Change"] = merged_df["Close"] - merged_df["Open"]
merged_df["Price_Change_Direction"] = merged_df["Price_Change"].apply(lambda x: 1 if x > 0 else 0)

# Sentiment bucketing
def label_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

merged_df["Sentiment_Category"] = merged_df["Sentiment"].apply(label_sentiment)

# Group by sentiment and price direction
grouped = merged_df.groupby(["Sentiment_Category", "Price_Change_Direction"]).size().unstack().fillna(0)
grouped["Total"] = grouped.sum(axis=1)
grouped["% Up"] = (grouped[1] / grouped["Total"]) * 100

print(grouped)

#Visualization Bucket Analysis

# Set plot style
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

# Bar plot of % Up
sns.barplot(
    x=grouped.index,
    y=grouped["% Up"].values,
    palette="viridis"
)

# Add title and labels
plt.title("Percentage of Upward Price Movement by Sentiment Category", fontsize=14)
plt.ylabel("Percentage of Up Days")
plt.xlabel("Sentiment Category")
plt.ylim(0, 100)

# Show percentage labels on bars
for index, value in enumerate(grouped["% Up"].values):
    plt.text(index, value + 1, f"{value:.1f}%", ha='center')

plt.tight_layout()
plt.show()

#********************************************************************
# AGGREGTE SENTIMENT VS SAME DAY STOCK MOVEMENT BY INDIVIDUAL TICKERS

# Load data
tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Preprocess
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Clean tweets and calculate sentiment
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"http\S+|[^a-zA-Z\s]", "", x))
tweets_df["Sentiment"] = tweets_df["Tweet"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Extract merge keys
tweets_df["Tweet_Date"] = tweets_df["Date"].dt.date
stock_df["Stock_Date"] = stock_df["Date"].dt.date

# Aggregate sentiment by day and stock
daily_sentiment = tweets_df.groupby(["Stock Name", "Tweet_Date"]).agg(
    Avg_Sentiment=("Sentiment", "mean"),
    Tweet_Count=("Sentiment", "count")
).reset_index()

# Merge sentiment with **same-day** stock prices
merged_daily = pd.merge(
    daily_sentiment,
    stock_df,
    left_on=["Stock Name", "Tweet_Date"],
    right_on=["Stock Name", "Stock_Date"]
)

# Calculate same-day price change and direction
merged_daily["Price_Change"] = merged_daily["Close"] - merged_daily["Open"]
merged_daily["Price_Change_Direction"] = merged_daily["Price_Change"].apply(lambda x: 1 if x > 0 else 0)

# Bucket sentiment into categories
def label_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

merged_daily["Sentiment_Category"] = merged_daily["Avg_Sentiment"].apply(label_sentiment)

# Group by stock and sentiment to calculate % Up
individual_ticker_groups = merged_daily.groupby(
    ["Stock Name", "Sentiment_Category", "Price_Change_Direction"]
).size().unstack().fillna(0)

individual_ticker_groups["Total"] = individual_ticker_groups.sum(axis=1)
individual_ticker_groups["% Up"] = (individual_ticker_groups[1] / individual_ticker_groups["Total"]) * 100

# Reset index and export
individual_ticker_groups = individual_ticker_groups.reset_index()
print(individual_ticker_groups)

# Save results
individual_ticker_groups.to_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/individual_sentiment_analysis_same_day.csv", index=False)

#********************************************************************
# AGGREGTE SENTIMENT VS NEXT DAY STOCK MOVEMENT BY INDIVIDUAL TICKERS

#Import

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")


# Preprocess
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Clean tweets and calculate sentiment
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"http\S+|[^a-zA-Z\s]", "", x))
tweets_df["Sentiment"] = tweets_df["Tweet"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Shift stock data to represent "next day" reaction
stock_df_shifted = stock_df.copy()
stock_df_shifted["Stock_Date"] = stock_df_shifted["Stock_Date"] - pd.Timedelta(days=1)

# Aggregate sentiment by day
daily_sentiment = tweets_df.groupby(["Stock Name", "Tweet_Date"]).agg(
    Avg_Sentiment=("Sentiment", "mean"),
    Tweet_Count=("Sentiment", "count")
).reset_index()

# Merge with next-day stock prices
merged_daily = pd.merge(
    daily_sentiment,
    stock_df_shifted,
    left_on=["Stock Name", "Tweet_Date"],
    right_on=["Stock Name", "Stock_Date"]
)

# Calculate price change and direction
merged_daily["Price_Change"] = merged_daily["Close"] - merged_daily["Open"]
merged_daily["Price_Change_Direction"] = merged_daily["Price_Change"].apply(lambda x: 1 if x > 0 else 0)

# Bucket sentiment
def label_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

merged_daily["Sentiment_Category"] = merged_daily["Avg_Sentiment"].apply(label_sentiment)

# Group by ticker + sentiment and calculate % Up
individual_ticker_groups = merged_daily.groupby(
    ["Stock Name", "Sentiment_Category", "Price_Change_Direction"]
).size().unstack().fillna(0)

individual_ticker_groups["Total"] = individual_ticker_groups.sum(axis=1)
individual_ticker_groups["% Up"] = (individual_ticker_groups[1] / individual_ticker_groups["Total"]) * 100

# Reset index for a clean DataFrame
individual_ticker_groups = individual_ticker_groups.reset_index()

# Display the result
print(individual_ticker_groups)

individual_ticker_groups.to_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/individual_sentiment_analysisWithTimeLag.csv", index=False)

#********************************************************************
# STOCK SPECIFIC SENTIMENT MODELS 
# LOGISTIC REGRESSION MODELS FOR EACH STOCK 
# USING ONLY DAULY SENTIMENT AND TWEET COUNT TO PREDICT NEXT-DAY PRICE DIRECTION


tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")


model_results = []

# Loop over each top-tweeted stock
for stock in merged_daily["Stock Name"].unique():
    stock_data = merged_daily[merged_daily["Stock Name"] == stock]
    
    # Features and target
    X = stock_data[["Avg_Sentiment", "Tweet_Count"]]
    y = stock_data["Price_Change_Direction"]

    # Ensure enough data points
    if len(stock_data) < 50 or len(np.unique(y)) < 2:
        continue
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    model_results.append({
        "Stock": stock,
        "Accuracy": accuracy,
        "Precision_Up": report["1"]["precision"],
        "Recall_Up": report["1"]["recall"],
        "F1_Up": report["1"]["f1-score"],
        "Support": report["1"]["support"]
    })

# Convert to DataFrame for display
model_results_df = pd.DataFrame(model_results).sort_values(by="Accuracy", ascending=False)



print(model_results_df)

# GROUP BAR CHART - LOGISTIC REGRESSION MODELS FOR EACH STOCK 

# Sort and select top N
top_n = 10
top_df = model_results_df.head(top_n)

# Set style
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot
metrics = ["Accuracy", "Precision_Up", "Recall_Up", "F1_Up"]
top_melted = top_df.melt(id_vars="Stock", value_vars=metrics, var_name="Metric", value_name="Score")

sns.barplot(data=top_melted, x="Stock", y="Score", hue="Metric", palette="viridis")

# Formatting
plt.title(f"Logistic Regression Performance by Stock (Top {top_n} by Accuracy)", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.legend(title="Metric")
plt.tight_layout()
plt.show()


#********************************************************************
# LOGISTIC REGRESSION MODELS FOR EACH STOCK 
# USING ONLY DAULY SENTIMENT AND TWEET COUNT TO PREDICT NEXT-DAY PRICE DIRECTION
# FEATURES USED TO ENHANCE MODEL:
# avg_Sentiment = Average sentiment score of tweets for a stick on a given day
# Tweet_Count = Number of tweets for a stock on a given day
# Price_Change_Prev_Day = Price change from the previous day for a stock
# day_of_week = The day of the week (0=Mon to 6= Sun) to capture weekday trading effects


tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Preprocess
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"http\S+|[^a-zA-Z\s]", "", x))
tweets_df["Sentiment"] = tweets_df["Tweet"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Filter to top-tweeted stocks
tweets_df["Tweet_Date"] = tweets_df["Date"].dt.date
stock_df["Stock_Date"] = stock_df["Date"].dt.date
top_stocks = tweets_df["Stock Name"].value_counts()
top_stocks = top_stocks[top_stocks > 1000].index.tolist()
tweets_df = tweets_df[tweets_df["Stock Name"].isin(top_stocks)]
stock_df = stock_df[stock_df["Stock Name"].isin(top_stocks)]

# Shift stock data for next-day prices
stock_df_shifted = stock_df.copy()
stock_df_shifted["Stock_Date"] = stock_df_shifted["Stock_Date"] - pd.Timedelta(days=1)

# Aggregate daily sentiment
daily_sentiment = tweets_df.groupby(["Stock Name", "Tweet_Date"]).agg(
    Avg_Sentiment=("Sentiment", "mean"),
    Tweet_Count=("Sentiment", "count")
).reset_index()

# Merge aggregated sentiment with stock data
merged_daily = pd.merge(
    daily_sentiment,
    stock_df_shifted,
    left_on=["Stock Name", "Tweet_Date"],
    right_on=["Stock Name", "Stock_Date"]
)

# Create target variable
merged_daily["Price_Change"] = merged_daily["Close"] - merged_daily["Open"]
merged_daily["Price_Change_Direction"] = merged_daily["Price_Change"].apply(lambda x: 1 if x > 0 else 0)

# Create enhanced features
merged_daily["Price_Change_Prev_Day"] = merged_daily.groupby("Stock Name")["Price_Change"].shift(1).fillna(0)
merged_daily["Day_Of_Week"] = pd.to_datetime(merged_daily["Tweet_Date"]).apply(lambda x: x.weekday())

# Train logistic regression models per stock
from sklearn.linear_model import LogisticRegression

results = []

for stock in merged_daily["Stock Name"].unique():
    stock_data = merged_daily[merged_daily["Stock Name"] == stock]

    X = stock_data[["Avg_Sentiment", "Tweet_Count", "Price_Change_Prev_Day", "Day_Of_Week"]]
    y = stock_data["Price_Change_Direction"]

    if len(stock_data) < 50 or len(np.unique(y)) < 2:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Stock": stock,
        "Accuracy": accuracy,
        "Precision_Up": report["1"]["precision"],
        "Recall_Up": report["1"]["recall"],
        "F1_Up": report["1"]["f1-score"],
        "Support": report["1"]["support"]
    })

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print(results_df)

# BAR CHART - LOGISTIC REGRESSION MODELS FOR EACH STOC
# Display top N stocks by accuracy
top_n = 10
top_enhanced = results_df.head(top_n)

# Melt for grouped bar plot
top_melted = top_enhanced.melt(id_vars="Stock", 
                               value_vars=["Accuracy", "Precision_Up", "Recall_Up", "F1_Up"], 
                               var_name="Metric", 
                               value_name="Score")

# Set up plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Bar plot
sns.barplot(data=top_melted, x="Stock", y="Score", hue="Metric", palette="mako")

# Customize
plt.title(f"Enhanced Logistic Regression Model Performance (Top {top_n} Stocks)", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.ylabel("Metric Score")
plt.xlabel("Stock")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()

# ********************************************************************
# RANDOM FOREST STOCK MODELS WITH SAME ENHANCED FEATURES

# Run Random Forest models with enhanced features

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

rf_model_results = []

for stock in merged_daily["Stock Name"].unique():
    stock_data = merged_daily[merged_daily["Stock Name"] == stock].copy()

    X = stock_data[["Avg_Sentiment", "Tweet_Count", "Price_Change_Prev_Day", "Day_Of_Week"]]
    y = stock_data["Price_Change_Direction"]

    if len(stock_data) < 50 or len(np.unique(y)) < 2:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    rf_model_results.append({
        "Stock": stock,
        "Accuracy": accuracy,
        "Precision_Up": report["1"]["precision"],
        "Recall_Up": report["1"]["recall"],
        "F1_Up": report["1"]["f1-score"],
        "Support": report["1"]["support"]
    })

# Display results
rf_model_df = pd.DataFrame(rf_model_results).sort_values(by="Accuracy", ascending=False)

print(rf_model_df)

# VISUALIZATION - RANDOM FOREST

# Select top N stocks by accuracy
top_n = 10
top_rf = rf_model_df.head(top_n)

# Melt for bar plot
top_rf_melted = top_rf.melt(id_vars="Stock", 
                            value_vars=["Accuracy", "Precision_Up", "Recall_Up", "F1_Up"], 
                            var_name="Metric", 
                            value_name="Score")

# Plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
sns.barplot(data=top_rf_melted, x="Stock", y="Score", hue="Metric", palette="Set2")

# Customize
plt.title(f"Random Forest Model Performance by Stock (Top {top_n})", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.ylabel("Metric Score")
plt.xlabel("Stock")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()

# ********************************************************************
# XGBoost Model

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")
# Run XGBoost models
xgb_model_results = []

for stock in merged_daily["Stock Name"].unique():
    stock_data = merged_daily[merged_daily["Stock Name"] == stock].copy()

    X = stock_data[["Avg_Sentiment", "Tweet_Count", "Price_Change_Prev_Day", "Day_Of_Week"]]
    y = stock_data["Price_Change_Direction"]

    if len(stock_data) < 50 or len(np.unique(y)) < 2:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    xgb_model_results.append({
        "Stock": stock,
        "Accuracy": accuracy,
        "Precision_Up": report["1"]["precision"],
        "Recall_Up": report["1"]["recall"],
        "F1_Up": report["1"]["f1-score"],
        "Support": report["1"]["support"]
    })

# Display results
xgb_model_df = pd.DataFrame(xgb_model_results).sort_values(by="Accuracy", ascending=False)

print(xgb_model_df)

# VISUALIZATION - XG BOOST MODEL
# Select top N performing stocks
top_n = 10
top_xgb = xgb_model_df.head(top_n)

# Melt for bar plot
top_xgb_melted = top_xgb.melt(id_vars="Stock", 
                              value_vars=["Accuracy", "Precision_Up", "Recall_Up", "F1_Up"], 
                              var_name="Metric", 
                              value_name="Score")

# Plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
sns.barplot(data=top_xgb_melted, x="Stock", y="Score", hue="Metric", palette="coolwarm")

# Customize
plt.title(f"XGBoost Model Performance by Stock (Top {top_n})", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.ylabel("Metric Score")
plt.xlabel("Stock")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()


# ENHANCED DATASET WITH MANUAL TECHNICAL FEATURES
# Load data
tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")

# Preprocessing
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
tweets_df["Tweet_Date"] = tweets_df["Date"].dt.date
stock_df["Stock_Date"] = stock_df["Date"].dt.date

tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"http\S+|[^a-zA-Z\s]", "", x))
tweets_df["Sentiment"] = tweets_df["Tweet"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Top stocks
top_stocks = tweets_df["Stock Name"].value_counts()
top_stocks = top_stocks[top_stocks > 1000].index.tolist()
tweets_df = tweets_df[tweets_df["Stock Name"].isin(top_stocks)]
stock_df = stock_df[stock_df["Stock Name"].isin(top_stocks)]

# Shift stock dates
stock_df_shifted = stock_df.copy()
stock_df_shifted["Stock_Date"] = stock_df_shifted["Stock_Date"] - pd.Timedelta(days=1)

# Aggregate sentiment
daily_sentiment = tweets_df.groupby(["Stock Name", "Tweet_Date"]).agg(
    Avg_Sentiment=("Sentiment", "mean"),
    Tweet_Count=("Sentiment", "count")
).reset_index()

# Merge
merged_daily = pd.merge(
    daily_sentiment,
    stock_df_shifted,
    left_on=["Stock Name", "Tweet_Date"],
    right_on=["Stock Name", "Stock_Date"]
)

# Target and Features
merged_daily["Price_Change"] = merged_daily["Close"] - merged_daily["Open"]
merged_daily["Price_Change_Direction"] = (merged_daily["Price_Change"] > 0).astype(int)
merged_daily["Price_Change_Prev_Day"] = merged_daily.groupby("Stock Name")["Price_Change"].shift(1).fillna(0)
merged_daily["Day_Of_Week"] = pd.to_datetime(merged_daily["Tweet_Date"]).apply(lambda x: x.weekday())

# Additional manually computed features
merged_daily["Return"] = merged_daily.groupby("Stock Name")["Close"].pct_change().fillna(0)
merged_daily["Volatility"] = (
    merged_daily.groupby("Stock Name")["Return"].rolling(window=5).std().reset_index(level=0, drop=True).fillna(0)
)
merged_daily["SMA_5"] = merged_daily.groupby("Stock Name")["Close"].transform(lambda x: x.rolling(window=5).mean())
merged_daily["SMA_10"] = merged_daily.groupby("Stock Name")["Close"].transform(lambda x: x.rolling(window=10).mean())
merged_daily["SMA_Diff"] = merged_daily["SMA_5"] - merged_daily["SMA_10"]

# Clean final dataset
merged_daily = merged_daily.dropna()

merged_daily.to_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/Enhanced_Dataset_W_Mannual_Technical_Features.csv", index=False)




#XGBOOST MODEL TO IMPROVE APPL

tweets_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_tweets.csv")
stock_df = pd.read_csv("/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/stock_yfinance_data.csv")


# === Preprocess Tweets ===
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True)
tweets_df["Tweet_Date"] = tweets_df["Date"].dt.date
tweets_df["Tweet"] = tweets_df["Tweet"].str.lower().apply(lambda x: re.sub(r"http\S+|[^a-zA-Z\s]", "", x))
tweets_df["Sentiment"] = tweets_df["Tweet"].apply(lambda x: TextBlob(x).sentiment.polarity)

# === Preprocess Stock Data ===
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
stock_df["Stock_Date"] = stock_df["Date"].dt.date

# === Filter Top-Tweeted Stocks ===
top_stocks = tweets_df["Stock Name"].value_counts()
top_stocks = top_stocks[top_stocks > 1000].index.tolist()
tweets_df = tweets_df[tweets_df["Stock Name"].isin(top_stocks)]
stock_df = stock_df[stock_df["Stock Name"].isin(top_stocks)]

# === Shift Stock Dates for Next-Day Reaction ===
stock_df_shifted = stock_df.copy()
stock_df_shifted["Stock_Date"] = stock_df_shifted["Stock_Date"] - pd.Timedelta(days=1)

# === Aggregate Daily Sentiment ===
daily_sentiment = tweets_df.groupby(["Stock Name", "Tweet_Date"]).agg(
    Avg_Sentiment=("Sentiment", "mean"),
    Tweet_Count=("Sentiment", "count")
).reset_index()

# === Merge Data ===
merged_daily = pd.merge(
    daily_sentiment,
    stock_df_shifted,
    left_on=["Stock Name", "Tweet_Date"],
    right_on=["Stock Name", "Stock_Date"]
)

# === Feature Engineering ===
merged_daily["Price_Change"] = merged_daily["Close"] - merged_daily["Open"]
merged_daily["Price_Change_Direction"] = (merged_daily["Price_Change"] > 0).astype(int)
merged_daily["Price_Change_Prev_Day"] = merged_daily.groupby("Stock Name")["Price_Change"].shift(1).fillna(0)
merged_daily["Day_Of_Week"] = pd.to_datetime(merged_daily["Tweet_Date"]).apply(lambda x: x.weekday())
merged_daily["Return"] = merged_daily.groupby("Stock Name")["Close"].pct_change().fillna(0)
merged_daily["Volatility"] = (
    merged_daily.groupby("Stock Name")["Return"].rolling(window=5).std()
    .reset_index(level=0, drop=True).fillna(0)
)

# === Clean Dataset ===
merged_daily = merged_daily.dropna()

# === Filter for AAPL ===
aapl_data = merged_daily[merged_daily["Stock Name"] == "AAPL"].copy()

# === Define Features and Target ===
features = [
    "Avg_Sentiment", "Tweet_Count", "Price_Change_Prev_Day",
    "Day_Of_Week", "Return", "Volatility"
]
X = aapl_data[features]
y = aapl_data["Price_Change_Direction"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Train XGBoost Model ===
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# === Predict and Evaluate ===
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# === Results Summary ===
results = {
    "Stock": "AAPL",
    "Accuracy": accuracy,
    "Precision_Up": report["1"]["precision"],
    "Recall_Up": report["1"]["recall"],
    "F1_Up": report["1"]["f1-score"],
    "Support": report["1"]["support"]
}

print("XGBoost Results for AAPL:")
print(results)

#VISUALIZATION - XGBoost Model APPLE

# Use the `results` dictionary from your model
aapl_results = {
    "Accuracy": results["Accuracy"],
    "Precision (Up)": results["Precision_Up"],
    "Recall (Up)": results["Recall_Up"],
    "F1 Score (Up)": results["F1_Up"]
}

# Prepare data
metrics = list(aapl_results.keys())
scores = list(aapl_results.values())

# Plot
plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")
bars = plt.barh(metrics, scores, color=sns.color_palette("deep"))

# Annotate values
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.2f}", va='center')

# Format
plt.xlim(0, 1)
plt.title("XGBoost Performance Metrics for AAPL", fontsize=14)
plt.xlabel("Score")
plt.tight_layout()
plt.show()

# Enhanced XGBOOST TESTLA With Enhanced Dataset Containing:
# AVG Sentiment, Tweet Count, Price_Change_Prev_Day, Day_of_Week, Return, Volatility

# Reload dataset
file_path = "/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/Enhanced_Dataset_W_Mannual_Technical_Features.csv"
merged_daily = pd.read_csv(file_path)

# Filter for TSLA
tsla_data = merged_daily[merged_daily["Stock Name"] == "TSLA"].copy()

# Define features and target
X = tsla_data[[
    "Avg_Sentiment", "Tweet_Count", "Price_Change_Prev_Day",
    "Day_Of_Week", "Return", "Volatility"
]]
y = tsla_data["Price_Change_Direction"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train XGBoost model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=30,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Results summary
tsla_results = {
    "Stock": "TSLA",
    "Accuracy": accuracy,
    "Precision_Up": report["1"]["precision"],
    "Recall_Up": report["1"]["recall"],
    "F1_Up": report["1"]["f1-score"],
    "Support": report["1"]["support"]
}
tsla_results

# VISUALIZATION - ENHANCED XGBOOST TESLA
# Use the `results` dictionary from your model

# Prepare TSLA results
tsla_metrics = {
    "Accuracy": tsla_results["Accuracy"],
    "Precision (Up)": tsla_results["Precision_Up"],
    "Recall (Up)": tsla_results["Recall_Up"],
    "F1 Score (Up)": tsla_results["F1_Up"]
}

# Plot setup
metrics = list(tsla_metrics.keys())
scores = list(tsla_metrics.values())

plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")

bars = plt.barh(metrics, scores, color=sns.color_palette("muted"))

# Annotate bars
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.2f}", va='center')

# Format plot
plt.xlim(0, 1)
plt.title("XGBoost Model Performance for TSLA", fontsize=14)
plt.xlabel("Score")
plt.tight_layout()
plt.show()



# EHANCED XGBOOST MODEL For AMD W Enhanced Dataset
file_path = "/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/Enhanced_Dataset_W_Mannual_Technical_Features.csv"
merged_daily = pd.read_csv(file_path)

# Filter for TSLA
amd_data = merged_daily[merged_daily["Stock Name"] == "AMD"].copy()

# Define features and target
X = amd_data[[
    "Avg_Sentiment", "Tweet_Count", "Price_Change_Prev_Day",
    "Day_Of_Week", "Return", "Volatility"
]]
y = amd_data["Price_Change_Direction"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train XGBoost model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=30,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Results summary
amd_results = {
    "Stock": "AMD",
    "Accuracy": accuracy,
    "Precision_Up": report["1"]["precision"],
    "Recall_Up": report["1"]["recall"],
    "F1_Up": report["1"]["f1-score"],
    "Support": report["1"]["support"]
}
amd_results

#VISUALIZATION - XGBOOST ENHANCED AMD

# Prepare AMD results
amd_metrics = {
    "Accuracy": amd_results["Accuracy"],
    "Precision (Up)": amd_results["Precision_Up"],
    "Recall (Up)": amd_results["Recall_Up"],
    "F1 Score (Up)": amd_results["F1_Up"]
}

# Set up plot
metrics = list(amd_metrics.keys())
scores = list(amd_metrics.values())

plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")

bars = plt.barh(metrics, scores, color=sns.color_palette("pastel"))

# Annotate bars
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.2f}", va='center')

# Format
plt.xlim(0, 1)
plt.title("XGBoost Model Performance for AMD", fontsize=14)
plt.xlabel("Score")
plt.tight_layout()
plt.show()


#EHANCED XGBOOST MODEL For AMZN W Enhanced Dataset

file_path = "/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/Enhanced_Dataset_W_Mannual_Technical_Features.csv"
merged_daily = pd.read_csv(file_path)

# Filter for TSLA
amzn_data = merged_daily[merged_daily["Stock Name"] == "AMZN"].copy()

# Define features and target
X = amzn_data[[
    "Avg_Sentiment", "Tweet_Count", "Price_Change_Prev_Day",
    "Day_Of_Week", "Return", "Volatility"
]]
y = amzn_data["Price_Change_Direction"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train XGBoost model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=30,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Results summary
amzn_results = {
    "Stock": "AMZN",
    "Accuracy": accuracy,
    "Precision_Up": report["1"]["precision"],
    "Recall_Up": report["1"]["recall"],
    "F1_Up": report["1"]["f1-score"],
    "Support": report["1"]["support"]
}
amzn_results

# VISUALIZATION ENHANCED XGBOOST - AMAZON 

# Prepare AMZN results
amzn_metrics = {
    "Accuracy": amzn_results["Accuracy"],
    "Precision (Up)": amzn_results["Precision_Up"],
    "Recall (Up)": amzn_results["Recall_Up"],
    "F1 Score (Up)": amzn_results["F1_Up"]
}

# Set up plot
metrics = list(amzn_metrics.keys())
scores = list(amzn_metrics.values())

plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")

bars = plt.barh(metrics, scores, color=sns.color_palette("light:#5A9", as_cmap=False))

# Annotate bars
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.2f}", va='center')

# Format
plt.xlim(0, 1)
plt.title("XGBoost Model Performance for AMZN", fontsize=14)
plt.xlabel("Score")
plt.tight_layout()
plt.show()

#EHANCED XGBoost Model for META

file_path = "/Users/alejandragonzalez/Library/CloudStorage/OneDrive-RockhurstUniversity/IndependentResearch/Enhanced_Dataset_W_Mannual_Technical_Features.csv"
merged_daily = pd.read_csv(file_path)

# Filter for TSLA
meta_data = merged_daily[merged_daily["Stock Name"] == "META"].copy()

# Define features and target
X = meta_data[[
    "Avg_Sentiment", "Tweet_Count", "Price_Change_Prev_Day",
    "Day_Of_Week", "Return", "Volatility"
]]
y = meta_data["Price_Change_Direction"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train XGBoost model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=30,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Results summary
meta_results = {
    "Stock": "META",
    "Accuracy": accuracy,
    "Precision_Up": report["1"]["precision"],
    "Recall_Up": report["1"]["recall"],
    "F1_Up": report["1"]["f1-score"],
    "Support": report["1"]["support"]
}
meta_results

# VISUALIZATION Enhanced XGBoost - Meta
# Prepare META results
meta_metrics = {
    "Accuracy": meta_results["Accuracy"],
    "Precision (Up)": meta_results["Precision_Up"],
    "Recall (Up)": meta_results["Recall_Up"],
    "F1 Score (Up)": meta_results["F1_Up"]
}

# Set up plot
metrics = list(meta_metrics.keys())
scores = list(meta_metrics.values())

plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")

bars = plt.barh(metrics, scores, color=sns.color_palette("muted"))

# Annotate bars
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.2f}", va='center')

# Format
plt.xlim(0, 1)
plt.title("XGBoost Model Performance for META", fontsize=14)
plt.xlabel("Score")
plt.tight_layout()
plt.show()

























