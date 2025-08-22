import re
import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep


import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud

import google.generativeai as genai


# ✅ Initialize WebDriver
def initialize_driver():
    options = Options()
    options.add_argument("--headless")  # run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    return webdriver.Chrome(options=options)

# ✅ Scrape product details
def scrape_product_details(url):
    driver = initialize_driver()
    driver.get(url)
    
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        product_name = driver.find_element(By.CSS_SELECTOR, 'a.VJA3rP').get_attribute("title")
        product_price = driver.find_element(By.XPATH, "//div[contains(text(), '₹')]").text
        image_element = driver.find_element(By.XPATH, "//img[contains(@src, 'rukminim')]")
        image_url = image_element.get_attribute("src")
    except:
        product_name, product_price, image_url = "Not Found", "Not Found", "Not Found"
    
    driver.quit()
    return product_name, product_price, image_url

# ✅ Scrape reviews from all categories (Most Helpful, Negative First, Positive First)
def scrape_reviews(url, num_pages=10):
    driver = initialize_driver()
    
    # Modify URL for reviews page
    base_url = url.replace("/p/", "/product-reviews/")
    base_url = re.sub(r"&sortOrder=.*", "", base_url)  # Remove any existing sorting parameters

    # Sorting categories
    categories = [
        f"{base_url}&aid=overall&certifiedBuyer=false&sortOrder=MOST_HELPFUL",  # Most Helpful
        f"{base_url}&aid=overall&certifiedBuyer=false&sortOrder=NEGATIVE_FIRST",  # Negative First
        f"{base_url}&aid=overall&certifiedBuyer=false&sortOrder=POSITIVE_FIRST"   # Positive First
    ]

    all_ratings, all_titles, all_comments = [], [], []

    # Loop through each category and scrape reviews
    for category_url in categories:
        for i in range(1, num_pages + 1):
            print(f"Scraping Page {i} of {category_url.split('&sortOrder=')[-1]}...")

            # Construct URL for each page
            page_url = f"{category_url}&page={i}"

            try:
                driver.get(page_url)
                sleep(2)  # Allow time for page load

                # Wait for reviews to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.XQDdHH.Ga3i8K, div.cDye2S'))
                )

                # Extract Ratings
                page_ratings = [ra.text for ra in driver.find_elements(By.CSS_SELECTOR, 'div.XQDdHH.Ga3i8K, div.cDye2S')]

                # Extract Review Titles
                page_titles = [re.text for re in driver.find_elements(By.CSS_SELECTOR, 'p.z9E0IG')]

                # Extract Review Comments
                page_comments = [co.text for co in driver.find_elements(By.CSS_SELECTOR, 'div.ZmyHeo')]

                # Ensure all lists have the same length
                min_len = min(len(page_ratings), len(page_titles), len(page_comments))

                print(f"✅ Found: {len(page_ratings)} Ratings, {len(page_titles)} Titles, {len(page_comments)} Comments")

                # Normalize list sizes (fill missing values with "N/A")
                max_length = max(len(page_ratings), len(page_titles), len(page_comments))
                page_ratings += ["N/A"] * (max_length - len(page_ratings))
                page_titles += ["N/A"] * (max_length - len(page_titles))
                page_comments += ["N/A"] * (max_length - len(page_comments))

                # Append to master lists
                all_ratings.extend(page_ratings)
                all_titles.extend(page_titles)
                all_comments.extend(page_comments)

                sleep(2)  # Prevent request blocking

            except Exception as e:
                print(f"⚠️ Error fetching page {i}: {e}")
                continue  # Skip this page and move to the next one

    driver.quit()

    # ✅ Ensure all lists are of equal length
    min_len = min(len(all_ratings), len(all_titles), len(all_comments))
    reviews_df = pd.DataFrame({
        "Rating": all_ratings[:min_len],
        "Title": all_titles[:min_len],
        "Review": all_comments[:min_len]
    })

    return reviews_df


# ✅ Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ✅ Sentiment classification
def classify_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.2:
        return "Positive"
    elif analysis.sentiment.polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

# ✅ Calculate sentiment rating
def calculate_sentiment_score(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.5:
        return 5
    elif polarity > 0.2:
        return 4
    elif polarity > -0.2:
        return 3
    elif polarity > -0.5:
        return 2
    else:
        return 1

# ✅ Calculate overall sentiment rating
def calculate_overall_rating(sentiment_ratings):
    return round(sentiment_ratings.mean(), 2)

# ✅ Predict overall sentiment
def predict_rating_category(rating):
    if rating >= 4:
        return "Good"
    elif 2 <= rating < 4:
        return "Okay"
    else:
        return "Bad"

# ✅ Generate bar graphs
def generate_bar_chart(data, labels, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=data, ax=ax, hue=labels, dodge=False, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    return img_buffer

# ✅ Generate word cloud
def generate_wordcloud_image(text):
    if not text.strip():
        return None  # Return None if there's no text to process
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    img_buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    return img_buffer

def summarize_reviews(reviews_df):
    genai.configure(api_key="AIzaSyBnaTFCYknijGaV9SaqiX1HdIrCWUXoFLM")
    all_reviews = "\n".join(reviews_df["Review"].dropna().tolist())
    prompt = f"""
    Summarize the following customer reviews in the **exact format** below:

    ---
    **Overall Summary:** (Start with an overview of general sentiment, highlighting the most common praises and complaints.)

    Specific positive points include:
    * Camera: (Mention user feedback on the camera.)
    * Display: (Describe the screen quality and user opinions.)
    * Performance: (Mention performance-related feedback, including smoothness and RAM.)
    * Design: (Summarize design-related aspects such as weight, aesthetics, and material quality.)
    * Value: (Talk about whether users feel the phone is worth the price.)

    Specific negative points include:
    * Heating: (Discuss any reports of heating issues.)
    * Battery: (Mention battery life complaints or praises.)
    * Software: (Describe any bugs, glitches, or missing features.)
    * Durability: (Mention any hardware issues or build quality concerns.)

    ---
    Now, summarize the following customer reviews accordingly:

    {all_reviews}
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text if response.text else "Summary not available."

#Main Function Call
def process_reviews(url):
    product_name, product_price, image_url = scrape_product_details(url)
    reviews_df = scrape_reviews(url)
    
    if reviews_df.empty:
        return {
            "image_url": image_url,
            "product_name": product_name,
            "product_price": product_price,
            "buy_link": url,
            "sentiment_rating": "N/A",
            "product_category": "Not enough data",
            "summary": "No reviews available.",
            "wordcloud_positive": None,
            "wordcloud_neutral": None,
            "wordcloud_negative": None,
            "sentiment_chart": None,
            "rating_chart": None
        }
    
    reviews_df.dropna(inplace=True)
    reviews_df["Full_Text"] = reviews_df["Title"] + " " + reviews_df["Review"]
    reviews_df["Full_Text"] = reviews_df["Full_Text"].apply(preprocess_text)
    reviews_df["Sentiment"] = reviews_df["Full_Text"].apply(classify_sentiment)
    reviews_df["SentimentRating"] = reviews_df["Full_Text"].apply(calculate_sentiment_score)
    
    overall_rating = calculate_overall_rating(reviews_df["SentimentRating"])
    overall_sentiment = predict_rating_category(overall_rating)
    
    # ✅ Generate word cloud images
    wordcloud_positive = generate_wordcloud_image(" ".join(reviews_df[reviews_df["Sentiment"] == "Positive"]["Full_Text"]))
    wordcloud_neutral = generate_wordcloud_image(" ".join(reviews_df[reviews_df["Sentiment"] == "Neutral"]["Full_Text"]))
    wordcloud_negative = generate_wordcloud_image(" ".join(reviews_df[reviews_df["Sentiment"] == "Negative"]["Full_Text"]))
    
    # ✅ Generate sentiment and ratings distribution graphs
    sentiment_counts = reviews_df["Sentiment"].value_counts()
    sentiment_chart = generate_bar_chart(sentiment_counts.values, sentiment_counts.index, "Sentiment Distribution", "Sentiment", "Count")
    
    rating_counts = reviews_df["Rating"].value_counts().sort_index()
    rating_chart = generate_bar_chart(rating_counts.values, rating_counts.index, "Ratings Distribution", "Rating", "Count")
    
    # ✅ Generate review summary
    summary = summarize_reviews(reviews_df)

    # ✅ Get Top 5 Reviews for Each Sentiment Category
    top_positive = reviews_df[reviews_df["Sentiment"] == "Positive"].head(5)[["Title", "Review"]]
    top_neutral = reviews_df[reviews_df["Sentiment"] == "Neutral"].head(5)[["Title", "Review"]]
    top_negative = reviews_df[reviews_df["Sentiment"] == "Negative"].head(5)[["Title", "Review"]]

    # Convert DataFrames to lists of dictionaries (for Streamlit display)
    top_positive = top_positive.to_dict(orient="records")
    top_neutral = top_neutral.to_dict(orient="records")
    top_negative = top_negative.to_dict(orient="records")

    
    return {
        "image_url": image_url,
        "product_name": product_name,
        "product_price": product_price,
        "buy_link": url,
        "sentiment_rating": overall_rating,
        "product_category": overall_sentiment,
        "summary": summary,
        "wordcloud_positive": wordcloud_positive,
        "wordcloud_neutral": wordcloud_neutral,
        "wordcloud_negative": wordcloud_negative,
        "sentiment_chart": sentiment_chart,
        "rating_chart": rating_chart,
        "top_positive": top_positive,
        "top_neutral": top_neutral,
        "top_negative": top_negative
    }
