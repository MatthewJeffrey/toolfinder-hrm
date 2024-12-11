import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os

# Set the page layout to wide mode and apply a title
st.set_page_config(layout="wide", page_title="Tool Finder for Electric Vehicle")

# Load data from Excel file (two sheets: Word and Number)
url = 'https://raw.githubusercontent.com/fwidio/ToolFinder/main/Database%20Master.xlsx'
df_word = pd.read_excel(url, sheet_name='Word', usecols=['Category', 'Relevant Word', 'Bin Location','Part Number'])
df_number = pd.read_excel(url, sheet_name='Number', usecols=['Category', 'Number', 'Bin'])

# Define the path for the images (outside of the conditional blocks)
image_folder = r"C:\Users\matth\Downloads\Photo"

# Apply custom CSS for futuristic design
st.markdown(
    """
    <style>
    body {
        background-color: #001f3f;
        color: #cce7ff;
        font-family: 'Courier New', monospace;
    }

    h1, h2, h3 {
        color: #30393b;
        text-shadow: 0 0 8px #edfdff, 0 0 16px #edfdff;
    }

    div[data-testid="column"] {
        background: rgba(0, 36, 77, 0.8);
        padding: 20px;
        border-radius: 10px;
    }

    input[type="text"] {
        background-color: #001f3f;
        color: #cce7ff;
        border: 2px solid #00e0ff;
        border-radius: 5px;
    }
    button {
        background-color: #00e0ff;
        color: #001f3f;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
    }
    button:hover {
        background-color: #00c4cc;
        color: #fff;
        box-shadow: 0 0 12px #00e0ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit interface
st.title('🔍 Tool Finder for Electric Vehicles')

# Set up columns for side-by-side layout
col1, col2 = st.columns(2)

# Left column: Word-based search
with col1:
    st.subheader("Text-based Tool Finder")

    # Input text box
    input_text = st.text_input('Enter a description of the tool (text):', key="text_input")

    if st.button('Submit Text', key="text_submit"):
        # Prepare the data for training using word-based data
        X_word = df_word['Relevant Word']
        y_word = df_word['Category']
       
        # Create a pipeline with CountVectorizer and Naive Bayes model for word data
        model_word = make_pipeline(CountVectorizer(ngram_range=(1, 3)), MultinomialNB())
        model_word.fit(X_word, y_word)

        # Get probability estimates for all categories
        class_probabilities = model_word.predict_proba([input_text])[0]
        classes = model_word.classes_
       
        # Find the maximum probability
        max_probability = class_probabilities.max()

        # Check if the maximum probability is above the threshold
        if max_probability >= 0.1:
            # Find all categories with the maximum probability
            highest_prob_categories = [(cls, prob) for cls, prob in zip(classes, class_probabilities) if prob == max_probability]
           
            st.write(f"The tool might belong to the following category/categories:")
            for category, probability in highest_prob_categories:
                st.write(f"- **{category}**")
               
                # Lookup the Bin Location for the category
                bin_location_values_word = df_word.loc[df_word['Category'] == category, 'Bin Location'].values
                bin_location_word = bin_location_values_word[0] if bin_location_values_word.size > 0 else 'Unknown'
                st.write(f"Bin Location: **{bin_location_word}**")

                part_number_values_word = df_word.loc[df_word['Category'] == category, 'Part Number'].values
                part_number_word = part_number_values_word[0] if part_number_values_word.size > 0 else 'Unknown'
                st.write(f"Part Number: **{part_number_word}**")
               
                # Define the path for the image based on the category
                image_path_word = os.path.join(image_folder, f"{category.lower()}.jpg")
               
                # Check if the image exists and display it
                if os.path.exists(image_path_word):
                    st.image(image_path_word, caption=f"Image for {category}", use_column_width=True)
                else:
                    st.write(f"No image available for {category}.")
        else:
            st.write("Undefined")

# Right column: Number-based search
with col2:
    st.subheader("Number-based Tool Finder")

    # Input number box
    input_number = st.text_input('Enter a description of the tool (number):', key="number_input")

    # Submit button for number-based input
    if st.button('Submit Number', key="number_submit"):
        # Prepare the data for training using number-based data
        X_number = df_number['Number'].astype(str)  # Convert to string to work with CountVectorizer
        y_number = df_number['Category']
       
        # Create a pipeline with CountVectorizer and Naive Bayes model for number data
        model_number = make_pipeline(CountVectorizer(token_pattern=r'\b\w+\b'), MultinomialNB())
        model_number.fit(X_number, y_number)

        # Predict the category based on number input
        prediction_number = model_number.predict([input_number])
        predicted_category_number = prediction_number[0]

        # Lookup the Bin Location from the dataframe
        bin_location_values_number = df_number.loc[df_number['Category'] == predicted_category_number, 'Bin'].values
        bin_location_number = bin_location_values_number[0] if bin_location_values_number.size > 0 else 'Unknown'

        # Display the result for number prediction
        st.write(f'The tool you are looking for based on number might be: **{predicted_category_number}**')
        st.write(f'Bin Location: **{bin_location_number}**')

        # Define the path for the image based on the predicted category
        image_path_number = os.path.join(image_folder, f"{str(predicted_category_number).lower()}.jpg")

        # Check if the image exists and display it
        if os.path.exists(image_path_number):
            st.image(image_path_number, caption=f"Image for {predicted_category_number}", use_column_width=True)
        else:
            st.write("Image not found.")