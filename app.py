import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
from ml import process_reviews 
from io import BytesIO

# Streamlit UI
st.title("Product Review Sentiment Analysis")
st.write("Enter Flipkart product URL to analyze reviews:")

# User input for product URL
product_url = st.text_input("Flipkart Product URL")

if st.button("Analyze"):
    if product_url:
        # Process the URL (Assume process_reviews() handles scraping and ML analysis)
        with st.spinner("Fetching reviews... Please wait for a few moments."):
            product_data = process_reviews(product_url)

        if product_data:
            # Display product details
            st.image(product_data['image_url'], caption=product_data['product_name'])
            st.write(f"**Name:** {product_data['product_name']}")
            st.write(f"**Price:** {product_data['product_price']}")
            st.write("[Buy Now]({})".format(product_data['buy_link']))
            
            # Sentiment Verdict
            st.subheader("Verdict")
            st.write(f"**Overall Sentiment:** {product_data['product_category']}")
            st.write(f"**Sentiment Rating:** {product_data['sentiment_rating']}")
            
            # Summary
            st.subheader("Summary")
            st.write(product_data['summary'])
            
            # Graphs
            st.subheader("Graphs")
            
            # Sentiment distribution chart
            st.image(product_data['sentiment_chart'], caption="Sentiment Distribution")

            # Ratings distribution chart
            st.image(product_data['rating_chart'], caption="Ratings Distribution")


            # ✅ **Top 5 Reviews in Table Format**
            st.subheader("Top 5 Reviews")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("### Positive Reviews")
                positive_df = pd.DataFrame(product_data['top_positive'])
                st.dataframe(positive_df, use_container_width=True)

            with col2:
                st.write("### Neutral Reviews")
                neutral_df = pd.DataFrame(product_data['top_neutral'])
                st.dataframe(neutral_df, use_container_width=True)

            with col3:
                st.write("### Negative Reviews")
                negative_df = pd.DataFrame(product_data['top_negative'])
                st.dataframe(negative_df, use_container_width=True)
            

            # Word Clouds
            st.subheader("Word Clouds")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(product_data['wordcloud_positive'], caption="Positive Words")
            with col2:
                st.image(product_data['wordcloud_neutral'], caption="Neutral Words")
            with col3:
                st.image(product_data['wordcloud_negative'], caption="Negative Words")

        else:
            st.error("Failed to fetch product details. Please try again with a valid Flipkart URL.")
    else:
        st.warning("Please enter a Flipkart product URL.")