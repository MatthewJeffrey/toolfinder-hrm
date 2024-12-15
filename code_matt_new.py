import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os
import difflib

# Set the page layout to wide mode and apply a title
st.set_page_config(layout="wide", page_title="Tool Finder for Electric Vehicle")

# Load data from Excel file (two sheets: Word and Number)
url = 'https://raw.githubusercontent.com/MatthewJeffrey/toolfinder-hrm/main/Database%20Master%20(1).xlsx'
#url = "C:/Users/fwidio/Downloads/Database Master.xlsx"
df_word = pd.read_excel(url, sheet_name='Word', usecols=['Category', 'Relevant Word', 'Bin Location','Part Number'])
df_number = pd.read_excel(url, sheet_name='Number', usecols=['Category', 'Number', 'Bin'])

# Define the path for the images (outside of the conditional blocks)
image_folder = r"C:\Users\matth\Downloads\Photo"

# Apply custom CSS for futuristic design
st.markdown(
    """
    <style>
    body {
        background-color: #dcdcdc; /* Gray background */
        color: #000000; /* Black text */
        font-family: 'Courier New', monospace;
    }

    h1, h2, h3 {
        color: #000000; /* Black headings */
        text-shadow: 0 0 8px #d3d3d3, 0 0 16px #d3d3d3;
    }

    div[data-testid="column"] {
        background: rgba(128, 128, 128, 0.8); /* Semi-transparent gray background */
        padding: 20px;
        border-radius: 10px;
    }

    input[type="text"] {
        background-color: #dcdcdc; /* Gray input background */
        color: #000000; /* Black text */
        border: 2px solid #000000; /* Black border */
        border-radius: 5px;
    }
    button {
        background-color: #000000; /* Black button background */
        color: #ffffff; /* White text */
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
    }
    button:hover {
        background-color: #696969; /* Dim gray on hover */
        color: #ffffff; /* White text */
        box-shadow: 0 0 12px #a9a9a9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit interface
st.title('🔍 Tool Finder for Electric Vehicles')

# Set up columns for side-by-side layout
col1, col2 = st.columns(2)

# Create a list of unique words from the "Relevant Word" column
unique_words = set()
for phrase in df_word['Relevant Word']:
    for word in phrase.lower().split():
        unique_words.add(word)

with col1:
    st.subheader("Word-based Tool Finder")


    # Input text box
    input_text = st.text_input('Enter a description of the tool (text):', key="text_input")

    if st.button('Submit Text', key="text_submit"):
        # Preprocess the input text
        input_text_processed = input_text.lower().strip()

        # Check if the input text contains any word not in the unique_words list
        input_words = input_text_processed.split()
        if all(word not in unique_words for word in input_words):
            st.write("The tool category is unknown.")
        else:
            # Prepare the data for training using word-based data
            X_word = df_word['Relevant Word']
            y_word = df_word['Category']
            
            # Create a pipeline with TfidfVectorizer and Naive Bayes model for word data
            model_word = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.1))
            model_word.fit(X_word, y_word)

            # Get probability estimates for all categories
            class_probabilities = model_word.predict_proba([input_text_processed])[0]
            classes = model_word.classes_
            
            # Find the maximum probability
            max_probability = class_probabilities.max()

            # Define a threshold for unknown classification
            threshold = 0.1

            # Find all categories with the maximum probability
            highest_prob_categories = [(cls, prob) for cls, prob in zip(classes, class_probabilities) if prob == max_probability]
            
            if max_probability < threshold:
                st.write("The tool category is unknown.")
            elif highest_prob_categories:
                st.write(f"The tool might belong to the following category/categories:")
                for category, probability in highest_prob_categories:
                    st.write(f"- *{category}*")
                    
                    # Lookup the Bin Location for the category
                    bin_location_values_word = df_word.loc[df_word['Category'] == category, 'Bin Location'].values
                    bin_location_word = bin_location_values_word[0] if bin_location_values_word.size > 0 else 'Unknown'
                    st.write(f"Bin Location: *{bin_location_word}*")

                    part_number_values_word = df_word.loc[df_word['Category'] == category, 'Part Number'].values
                    part_number_word = part_number_values_word[0] if part_number_values_word.size > 0 else 'Unknown'
                    st.write(f"Part Number: *{part_number_word}*")
                    
                    # Define the path for the image based on the category
                    image_path_word = os.path.join(image_folder, f"{category.lower()}.jpg")
                    
                    # Check if the image exists and display it
                    if os.path.exists(image_path_word):
                        st.image(image_path_word, caption=f"Image for {category}", use_column_width=True)
                    else:
                        st.write(f"No image available for {category}.")
            else:
                st.write("No matching categories found.")
            
# Right column: Number-based search
with col2:
    st.subheader("Number-based Tool Finder")

    # Input number box
    input_number = st.text_input('Enter a description of the tool (number):', key="number_input")

    # Submit button for number-based input
    if st.button('Submit Number', key="number_submit"):
        # Find the exact match in the dataframe
        matched_row = df_number[df_number['Number'] == input_number]

        if not matched_row.empty:
            predicted_category_number = matched_row['Category'].values[0]
            bin_location_number = matched_row['Bin'].values[0]

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
        else:
            # Find the closest match using difflib
            closest_matches = difflib.get_close_matches(input_number, df_number['Number'].astype(str), n=1, cutoff=0.1)
            if closest_matches:
                closest_match = closest_matches[0]
                matched_row = df_number[df_number['Number'] == closest_match]
                predicted_category_number = matched_row['Category'].values[0]
                bin_location_number = matched_row['Bin'].values[0]

                # Display the result for the closest match
                st.write(f'Exact match not found. The closest match is: **{closest_match}**')
                st.write(f'The tool you are looking for based on number might be: **{predicted_category_number}**')
                st.write(f'Bin Location: **{bin_location_number}**')

                # Define the path for the image based on the predicted category
                image_path_number = os.path.join(image_folder, f"{str(predicted_category_number).lower()}.jpg")

                # Check if the image exists and display it
                if os.path.exists(image_path_number):
                    st.image(image_path_number, caption=f"Image for {predicted_category_number}", use_column_width=True)
                else:
                    st.write("Image not found.")
            else:
                st.write("Unknown")
