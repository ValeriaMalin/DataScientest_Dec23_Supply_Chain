#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import emoji
import demoji
import nltk
from nltk.corpus import stopwords

# Load the preprocessed data
df = pd.read_csv('/Users/solozobovavaleria/Supply_Chain_cleaned.csv')

st.title("Supply Chain : Sentiment Analysis Project")
st.sidebar.title("Table of contents")
pages=["Our Project","Data Processing", "Wordclouds", "Modelling", "Conclusion", "About"]
page=st.sidebar.radio("Go to", pages)


if page == pages[0] : 
    st.write("### Our Project")
    st.write(
    "With our project we wanted to gain valuable insights from cellphone reviews of  Amazon, providing a deeper understanding of the products strengths and weaknesses and to improve the Supply Chain.",
    "We were exploring the following steps:\n\n"
    "1. **Sentiment Classification:** Classifying the sentiment of reviews as positive (4-5 stars), negative (1-2 stars), or neutral (3 stars) by  the stars given.\n"
    "2. **Thematic Analysis:** Discovering issues that contribute to the positive, negative, or neutral classification of reviews by analyzing them trough Word Clouds.\n"
    "3. **Brand Comparison:** Investigating whether these issues highlighted in reviews vary between products from the top reviewed brands, making our analysis more brand-specific.\n"
    "4. **Word2Vec:** Identifying and understanding the relationship between the most used words through mapping them as vectors\n\n"
    "5. **Prediction on Sentiment:** Identifying and understanding the relationship between the most used words through mapping them as vectors\n\n"
    "Through our Streamlit app, you'll have the opportunity to explore these objectives by visualize the data and predictiong your own model."
)


    st.write("We used a dataset from Kaggle. Click on the button below to check it out!")
    url = "https://www.kaggle.com/code/imdevskp/amazon-reviews-sentiment-analysis-and-prediction/input"
    st.link_button("Amazon Reviews Sentiment Analysis and Prediction", url, help=None, type="secondary", disabled=False, use_container_width=False)
    st.write(" This is our precrocessed Dataframe:\n\n")
    st.dataframe(df.head(30))


if page == pages[1] : 
    st.write("### Dataprocessing")
    st.write("### Focus of Dataset")
    
    st.write("To ensure a more targeted analysis, we narrowed our data down to cellphones only.",
         "This refinement allows us to extract insights directly relevant to the supply chain of mobile devices,",
         "eliminating unnecessary noise from cellphone accessories -therefor enhancing the significance of our findings.\n")

    st.write("### Dataprocessing")
    st.write("The initial dataset faced challenges with duplicated rows and missing values.",
         "We addressed this by removing unnecessary duplicates and deleting entries without reviews.",
         "For a large amount of unknown Brandnames, we traced their brandname from the Product Name."
         "We reduced inconsistent capitalization and formatting in the Brand Name column.",
         "Furthermore we deleted all cellphone accessories contained in the dataset to focus only on the cellphones.")

    st.write("Data preprocessing plays a crucial role in constructing effective Machine Learning models,"
        "as the quality of results is closely tied to the thoroughness of data preprocessing. "
        "Our text preprocessing pipeline involved several key steps:\n\n"
        "- **Emoji Handling:** Replacing emojis using the vocabulary from the demoji package, ensuring "
        "that emoticons are appropriately represented in the text.\n"
        "- **Digit Removal:** Eliminating numerical digits from the text, focusing on the textual content "
        "rather than numerical values.\n"
        "- **Exclamation Mark Replacement:** Substituting exclamation marks with the word 'exclamation' "
        "to capture their emotional significance in the text.\n"
        "- **Special Character Removal:** Removing other special characters to maintain a clean and "
        "standardized text format.\n"
        "- **Lowercasing:** Converting all words to lowercase to ensure uniformity and prevent "
        "discrepancies due to case variations.\n"
        "- **Stopword Removal:** Eliminating common stopwords, which were augmented with domain-specific "
        "terms from our reviews to enhance the relevance of the stopword list.\n"
        "- **Tokenization:** Splitting sentences into individual words (tokenization), a crucial step "
        "for further analysis and feature extraction.\n\n")

    st.write("#### An Example")
    import demoji
    import re
    import emoji
    import nltk
    from nltk.corpus import stopwords

    # Emoji Handling
    def replace_emojis(text):
        return ' '.join([demoji.replace_with_desc(emo) for emo in text.split()])

    # Digit Removal
    def remove_digits(text):
        return re.sub(r"\d+", "", text)

    # Exclamation Mark Replacement
    def replace_exclamation(text):
        return re.sub(r"!", " exclamation ", text)

    # Special Character Removal
    def remove_special_characters(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Lowercasing
    def convert_to_lowercase(text):
        return text.lower()

    # Stopword Removal
    nltk.download('stopwords')
    nltk.download('punkt')
    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        custom_stopwords = ['phone', 'phones', 'cell', 'amazon', 'review', 'reviews', 'product', 'products',
                            'buy', 'samsung', 'apple', 'android', 'mobile', 'galaxy', 'google',
                            'iphone', 'verizon', 'work', 'movistar', 'telefonica', 'lg', 'htc', 'maybe', 'blu',
                            'really', 'very', 'new', 'work', 'get', 'say']
        stop_words.update(custom_stopwords)
        
        return ' '.join([word for word in text.split() if word not in stop_words])

    # Tokenization
    def tokenize_text(text):
        return nltk.word_tokenize(text)

    # Example usage
    original_text = "This is a sample text with 😊, 5 stars, and some special characters! It's great!"
    st.write("**Here is an example:**\n",original_text)

    def text_processing(text):
        processed_text = replace_emojis(text)
        processed_text = remove_digits(processed_text)
        processed_text = replace_exclamation(processed_text)
        processed_text = remove_special_characters(processed_text)
        processed_text = convert_to_lowercase(processed_text)
        processed_text = remove_stopwords(processed_text)
        tokenized_text = tokenize_text(processed_text)

        # Display processed text and tokenized words
        st.write("**Processed Example:**\n", processed_text)
        st.write("**Tokenized Words of the given example:**\n", tokenized_text)

    text_processing(original_text)

    st.write("#### Create your own review")
    user_input = st.text_area("Enter your text:")
    text_processing(user_input)

if page == pages[2] : 
    st.write("###Wordclouds")

    # Distribution of Review Lengths
    st.header("Distribution of Review Lengths")
    # Display the image
    image_path = "/Users/solozobovavaleria/Review Length per Rating.png"
    st.image(image_path)


    # Separate df into positive, neutral and negative data
    df_neg = df[df["sentiment"] == "Negative"]
    df_pos = df[df["sentiment"] == "Positive"]
    df_neutral = df[df["sentiment"] == "Neutral"]


    # Example usage in Streamlit app
    st.title("Word Clouds")
    st.header("Positive Reviews")
    
    selected_sentiment = st.selectbox("Select Semtiment", ["Positive", "Negative", "Neutral"])
    f"Word Cloud for {selected_sentiment} Reviews"

    if selected_sentiment == "Positive":
        # Display the image
        image_path = "/Users/solozobovavaleria/PositiveWordCloud.png"
        st.image(image_path)

    if selected_sentiment == "Negative":
        # Display the image
        image_path = "/Users/solozobovavaleria/NegativeWordCloud.png"
        st.image(image_path)

    if selected_sentiment == "Neutral":
        # Display the image
        image_path = "/Users/solozobovavaleria/NeutralWordCloud.png"
        st.image(image_path)


    # Display Word Clouds for Samsung and Apple
    st.header("Brand and Sentiment Selection")
    '''
   In the graph presented below, we have aggregated data on the 10 most reviewed brands. To provide insights for weak points in the supply chain,
     we narrowed our focus to the three most reviewed brands: Samsung, Apple, and Blu. Subsequently, we generated Word Clouds
       tailored to each of these brands, categorizing reviews into positive, neutral, and negative sentiments.
     This detailed analysis offers a nuanced understanding of the areas where Samsung, Apple, and Blu can enhance their respective supply chains. '''
    
    # Display the image
    image_path = "/Users/solozobovavaleria/The Most reviewed.png"
    st.image(image_path)
    selected_brand = st.selectbox("Select Brand", ["Samsung", "Apple", "Blu"])

    # Display Word Cloud for Selected Brand and Sentiment
    f"Word Cloud for {selected_brand} Reviews"

     # Display selected DataFrame based on brand and sentiment
    if selected_brand == "Samsung":
        # Display the image
        image_path = "/Users/solozobovavaleria/SamsungWordCloud.png"
        st.image(image_path)
    elif selected_brand == "Apple":
        # Display the image
        image_path = "/Users/solozobovavaleria/AppleWordCloud.png"
        st.image(image_path)
    elif selected_brand == "Blu":
        #Display the image
        image_path = "/Users/solozobovavaleria/BluWordCloud.png"
        st.image(image_path)

if page == pages[3] :
    tab1, tab2, tab3, tab4 = st.tabs(["Multiclass Sentiment Classification", "Binary Sentiment Classification", "Multiclass Neural Networks", "Binary Neural Networks"])

    with tab1:
        st.write("### Modelling with Multiclass Sentiment Classification")
        st.write("X contains the text data (reviews), and y contains the corresponding sentiment labels (positive, negative, or neutral).",
        "The data is split into a training and test set using the train_test_split function from scikit-learn.",
        "It allocates 75%  the data to training and 25% to testing. We use the CountVectorizer to convert the text data into numerical vectors.",
        "It represents each single review as a bag-of-words, counting the occurrences of each word")
        
        import pickle
        import joblib
        from sklearn.metrics import confusion_matrix

        options = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting', 'XGBoost']
        option = st.selectbox('Choice of the model', options)
        st.write('The chosen model is:', option)
        
        # Load the trained model using Joblib
        model_filename = f"{option.lower().replace(' ', '_')}_model.joblib"
        clf = joblib.load(model_filename)

        # Load accuracy
        accuracy_filename = f"{option.lower().replace(' ', '_')}_accuracy.pkl"
        with open(accuracy_filename, 'rb') as accuracy_file:
            accuracy = pickle.load(accuracy_file)

        # Load confusion matrix
        confusion_matrix_filename = f"{option.lower().replace(' ', '_')}_confusion_matrix.pkl"
        with open(confusion_matrix_filename, 'rb') as confusion_matrix_file:
            confusion_matrix_result = pickle.load(confusion_matrix_file)

        display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix'))
        # Display the chosen information
        if display == 'Accuracy':
            st.write(f'Accuracy for {option}: {accuracy}')
        elif display == 'Confusion matrix':
            st.dataframe(confusion_matrix_result)

        '''We can see that the Random Forest model is performing best, therefor we want to improve the parameters with GridSearchCV.
        We create the following parameter grid:'''

        st.code("""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        """, language="python", line_numbers=False)
        '''However since the execution time with such a big a amount of text features exceeded the RAM
        capacity, we narrowed the data down to a sample fraction of 10% of the actual textdata. On this
        sample we performed the GridSearch to extract the following best parameters:'''

        st.code("""
        {'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 200
        }
        """, language="python", line_numbers=False)

        '''Even thou we were using these parameters for a better performance of the Random Forest Model it
        has not improved the outcome, since its performance on the neutral sentiment is worse (28%
        instead of 73%) and the accuracy dropped to 84% compared to 94% without the parameters.
        The default Random Forest turns out to be better.'''

        import os
        '''Since we are dealing with an imbalanced dataset we tried Random Oversampling and SMOTE in hopes for a better output.'''
        
        options = ['Random Oversampling', 'Random Undersampling', 'SMOTE', 'Cluster Centroids']
        option = st.selectbox('Choice of the model', options)
        st.write('The chosen model is:', option)

        display = st.radio('What do you want to show?', ('Classification Report', 'Confusion matrix'), key='unique_key_for_radio')

        # Define image filenames
        confusion_matrix_image = f"{option.lower().replace(' ', '_')}_confusionmatrix.png"
        classification_report_image = f"{option.lower().replace(' ', '_')}_classificationreport.png"

        # Define image paths
        confusion_matrix_path = os.path.join("/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction", confusion_matrix_image)
        classification_report_path = os.path.join("/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction", classification_report_image)

        # Display the chosen information
        if display == 'Classification Report':
            if os.path.exists(classification_report_path):
                st.image(classification_report_path)
            else:
                st.warning(f"Image not found: {classification_report_path}")
        elif display == 'Confusion matrix':
            if os.path.exists(confusion_matrix_path):
                st.image(confusion_matrix_path)
            else:
                st.warning(f"Image not found: {confusion_matrix_path}")
                
        '''We used the TfidfVectorizer to convert the text data into a TF-IDF (Term Frequency-Inverse Document Frequency) matrix.
        This matrix shows the importance of each word.'''
        
        img_path= ("/Users/paulabeck/Desktop/30mostimportantfeatures.png")
        st.image(img_path)
        
    with tab2:
        st.write("### Modelling with Binary Sentiment Classification")
        
        options_binary = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting', 'XGBoost']
        option_binary = st.selectbox('Choice of the model', options_binary, key="model_selection")
        st.write('The chosen model is:', option_binary)
        
        '''
        # Load the trained model using Joblib
        model_filename = f"{option_binary.lower().replace(' ', '_')}_binary_model.joblib"
        clf = joblib.load(model_filename)
        
        # Load accuracy
        accuracy_filename = f"{option_binary.lower().replace(' ', '_')}_accuracy_binary.pkl"
        with open(accuracy_filename, 'rb') as accuracy_file:
            accuracy = pickle.load(accuracy_file)

        # Load confusion matrix
        confusion_matrix_filename = f"{option_binary.lower().replace(' ', '_')}_confusion_matrix_binary.pkl"
        with open(confusion_matrix_filename, 'rb') as confusion_matrix_file:
            confusion_matrix_result = pickle.load(confusion_matrix_file)

        display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix'))
        # Display the chosen information
        if display == 'Accuracy':
            st.write(f'Accuracy for {option_binary}: {accuracy}')
        elif display == 'Confusion matrix':
            st.dataframe(confusion_matrix_result)
        '''
    with tab3:
        st.write("### Modelling with Neural Networks for Multiclass Sentiment Classification")
        
        st.write("Using Principal Component Analysis (PCA) we visualize the relationships between the word vectors in a Word2Vec model:",
                 'We defined pairs of words one aspect being words like "screen," "camera," "return," "quality," "problem," "service," "seller," and "battery"',
                 'and the other being positively or negatively associated words like "great" or "worst"')
        image_path = "/Users/paulabeck/Desktop/Word2vec.png"
        st.image(image_path)
        
        image_path = "/Users/paulabeck/Desktop/CNN_multi_Info.png"
        st.image(image_path)
        
        image_path = "/Users/paulabeck/Desktop/AccuracyCNN.png"
        st.image(image_path)
    with tab4:
        st.write("### Modelling with Neural Networks for Binary Sentiment Classification")
        
        image_path = "/Users/paulabeck/Desktop/AccuracyCNN_binary.png"
        st.image(image_path)
        
        
if page == pages[4] :
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Handle NaN values in the 'Text' column
    df = df.dropna(subset=['Text'])
    @st.cache(allow_output_mutation=True)
    def preprocess_data(df):
        # Splitting into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(df.Text, df.sentiment, test_size=0.25, random_state=42)
        
        # Using CountVectorizer to transform
        vectorizer = CountVectorizer()
        X_train_transformed = vectorizer.fit_transform(X_train)
        
        return X_train_transformed, y_train, vectorizer

    @st.cache(allow_output_mutation=True)
    def train_model(X_train_transformed, y_train):
        # Train the RandomForestClassifier model
        clf = RandomForestClassifier(n_jobs=-1)
        clf.fit(X_train_transformed, y_train)
        
        return clf
    st.title("Sentiment Prediction App")

# Load and preprocess the data
X_train_transformed, y_train, vectorizer = preprocess_data(df)

# Train the model
clf = train_model(X_train_transformed, y_train)

# Input for user review
user_review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    # Handle NaN values in the user input
    user_review = user_review if pd.notna(user_review) else ''

    # Transform user input using the pre-trained CountVectorizer
    user_input_transformed = vectorizer.transform([user_review])

    # Make prediction using the RandomForestClassifier model
    prediction = clf.predict(user_input_transformed)

    # Display the result
    st.subheader("Sentiment Prediction:")
    st.write(prediction[0])

    st.write("### How do we evaluate our models?")
    
    '''In summary we can say that the Random Forest model performs best in comparison to all the other models,
    with high precision, recall, and F1-score for Negative and Positive classes. Other than expected the GridSearchCV 
    for the best parameters of the Random Forest Model didn't improve the accuracy.
    In all the models we tried the Neutral class performed poorly -compared to the postive and negative sentiment-,
    so there is room for improvement in predicting the Neutral sentiment. This is due to the imbalanced dataset, meaning there are
    significantly fewer reviews of Neutral feedback (3 Stars given) compared to the other classes.
    Also the neutral sentiment is more challenging to detect accurately as it lacks strong sentiment or emotional tone. All our
    models struggled to differantiate between truly neutral expressions and those that contain subtle sentiments. 
    Therefor we decided to use Random Over- and Undersampling as well as SMOTE and
    Cluster Centroids to see wether we could further improve the accuracy by balancing out the dataset. Unfortunately, that did not 
    help either.'''
    
if page == pages[5] :
    
    st.write("### Contribution to the project")
    
    import webbrowser
    
    col1, col2 = st.columns([1, 2]) 
    image_path = "/Users/paulabeck/Desktop/logo-datascientest.png"
    col1.image(image_path)
   
    col2.markdown("This project was carried out as part of the Data Scientist training by DataScientest.com\n"
        "in cooperation with Sarbonne University.\n"
        "Our Project Manager was Axalia Levanchaud. \n #### Team Members:\n")

    
    # Valeriya Solozobova
    col2.markdown("Valeriya Solozobova")
    linkedin_url_valeriya = "https://www.linkedin.com/in/valeriya-solozobova/"
    linkedin_logo_html_valeriya = '<a href="{}" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png"/></a>'.format(linkedin_url_valeriya)
    col2.markdown(linkedin_logo_html_valeriya, unsafe_allow_html=True)
    
    # Paula Robina Beck
    col2.markdown("Paula Robina Beck")
    linkedin_url_paula = "https://www.linkedin.com/in/paula-robina-beck-b406a3172/"
    linkedin_logo_html_paula = '<a href="{}" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png"/></a>'.format(linkedin_url_paula)
    col2.markdown(linkedin_logo_html_paula, unsafe_allow_html=True)