import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Web scraping function that saves data into a CSV
def scrape_data(query):
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    driver = webdriver.Chrome()
    base_url = "https://www.amazon.in/s?k="
    target_url = f"{base_url}{query.replace(' ', '+')}"

    driver.get(target_url)
    time.sleep(2)

    products = driver.find_elements(By.CLASS_NAME, "s-result-item")

    # Data lists for CSV
    titles = []
    prices = []
    links = []

    for idx, product in enumerate(products):
        try:
            title = product.find_element(By.TAG_NAME, "h2").text
            price = product.find_element(By.CLASS_NAME, "a-price-whole").text
            link = product.find_element(By.TAG_NAME, "a").get_attribute("href")

            titles.append(title)
            prices.append(price)
            links.append(link)
        except Exception as e:
            continue

    # Create a DataFrame and save it to CSV
    df = pd.DataFrame({
        'Title': titles,
        'Price': prices,
        'Link': links
    })

    csv_file_path = os.path.join(data_dir, f"{query}_data.csv")
    df.to_csv(csv_file_path, index=False)

    driver.quit()
    return csv_file_path

# CSV query function using LangChain
def query_csv(csv_file, user_query):
    data = pd.read_csv(csv_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data, verbose=True)
    answer = agent.run(user_query)
    return answer

# Streamlit app layout
def main():
    st.set_page_config(page_title="Web Scraper & CSV Query Tool", layout="centered")

    # CSS for UI/UX
    st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .block-container { padding: 2rem 3rem; }
    .title { color: #4CAF50; font-size: 2rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='title'>Web Scraper & CSV Query Tool</h2>", unsafe_allow_html=True)
    
    st.write("### 1. Web Scraping Section")
    query = st.text_input("Enter a product to scrape (e.g., '2L water bottle')")
    if st.button("Scrape Data"):
        with st.spinner("Scraping data..."):
            csv_file_path = scrape_data(query)
            st.success(f"Data for '{query}' scraped successfully!")

            # Provide a link to download the CSV file
            with open(csv_file_path, "rb") as file:
                st.download_button(label="Download CSV file", data=file, file_name=f"{query}_data.csv", mime="text/csv")
    
    st.write("### 2. CSV Query Section")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if csv_file:
        user_query = st.text_input("Enter your query for the CSV")
        if user_query and st.button("Submit Query"):
            with st.spinner("Processing query..."):
                answer = query_csv(csv_file, user_query)
                st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
