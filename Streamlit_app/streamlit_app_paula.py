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

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font"></p>', unsafe_allow_html=True)

# Load the preprocessed data
df = pd.read_csv('/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction/Supply_Chain_preprocessed.csv')

st.title("Supply Chain: Sentiment Analysis for Amazon Reviews on Cellphones")
st.write("**A Project by Valeria Solozobova and Paula Beck**")
st.sidebar.title("Table of contents")
pages=["Our Project","Data Processing", "Wordclouds","Word2Vec for Supply Chain", "Modelling", "Conclusion", "About"]
page=st.sidebar.radio("Go to", pages)


if page == pages[0] : 
    st.write(
        "### Our Project"
    )
    st.write(
        "With our project we wanted to gain valuable insights from cellphone reviews of  Amazon, providing a deeper understanding of the products strengths and weaknesses and to improve the Supply Chain.",
    "We were exploring the following steps:\n\n")
    col1, col2, col3 = st.columns([1, 20, 2])
    with col2:
        st.write(
        "1. **Sentiment Classification:** Classifying the sentiment of reviews as positive (4-5 stars), negative (1-2 stars), or neutral (3 stars) by  the stars given.\n"
        "2. **Thematic Analysis:** Discovering issues that contribute to the positive, negative, or neutral classification of reviews by analyzing them trough Word Clouds.\n"
        "3. **Brand Comparison:** Investigating whether these issues highlighted in reviews vary between products from the top reviewed brands, making our analysis more brand-specific.\n"
        "4. **Word2Vec:** Identifying and understanding the relationship between the most used words through mapping them as vectors\n\n"
        "5. **Prediction on Sentiment:** Identifying and understanding the relationship between the most used words through mapping them as vectors\n\n"
        )

    st.write(
        "Through our Streamlit app, you'll have the opportunity to explore by visualizing the data and predicting your own model.",
        "We used a dataset from Kaggle. Click on the button below to check it out!"
        )
    url = "https://www.kaggle.com/code/imdevskp/amazon-reviews-sentiment-analysis-and-prediction/input"
    st.link_button("Amazon Reviews Sentiment Analysis and Prediction", url, help=None, type="secondary", disabled=False, use_container_width=False)
    st.write(" This is our precrocessed Dataframe:\n\n")
    st.dataframe(df.head(30))


if page == pages[1] : 
    st.write("### Exploratory Data Analysis and Primary Data Preprocessing")
    
    st.write("To ensure a more targeted analysis, we narrowed our data down to cellphones only.",
         "This refinement allows us to extract significant insights into the supply chain of mobile devices,",
         "eliminating unnecessary noise from cellphone accessories.\n")

    st.write("### Dataprocessing")
    st.write("The initial dataset faced challenges with duplicated rows and missing values.",
         "We removed unnecessary duplicates and deleting entries without reviews.",
         "For a large amount of unknown Brandnames, we traced their brandname from the Product Name."
         "We reduced inconsistent capitalization and formatting in the Brand Name column.",
         )
    col1, col2 = st.columns([1, 1])

    # Left column for the image
    with col1:
        image_path = "/Users/paulabeck/Desktop/pngs/Rating_Distribution.png"
        st.image(image_path, width=500)  

    # Right column for the explanation
    with col2:
        st.write("### Distribution of Ratings")
        st.write("""
        Our reviews are imbalanced dataset,  the distribution of stars given differs greatly. 
        That could lead to challenges in training machine learning models that can accurately 
        predict less common ratings. This is why we will use oversampling to deal with the imbalance.
        """)
    st.write("Data preprocessing plays a crucial role in constructing effective Machine Learning models."
            "Our text preprocessing pipeline involved several key steps:\n\n")
    
    col1, col2, col3 = st.columns([1, 20, 2])
    with col2:      
        st.write("- **Emoji Handling:** Replacing emojis using the vocabulary from the demoji package, ensuring "
            "that emoticons are written out in text.\n"
            "- **Digit Removal:** Eliminating numerical digits from the text, focusing only on the textual content\n"
            "- **Exclamation Mark Replacement:** Substituting exclamation marks with the word 'exclamation' "
            "to understand their emotional significance in the text.\n"
            "- **Special Character Removal:** Removing other special characters to maintain a clean and "
            "standardized format.\n"
            "- **Lowercasing:** Converting all words to lowercase to prevent"
            "discrepancies due to case variations.\n"
            "- **Stopword Removal:** Eliminating common stopwords(also domain-specific stopwords)  "
            "to enhance the interpretability of our wordclouds later on.\n"
            "- **Tokenization:** Splitting sentences into smaller units (tokens)"
            "to then use feature extraction.\n\n")

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
    # Distribution of Review Lengths
    st.write("### Distribution of Review Lengths per Stars given")
    # Display the image
    image_path = "/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction/Reviewlength.png"
    st.image(image_path, width = 600)


    # Separate df into positive, neutral and negative data
    df_neg = df[df["sentiment"] == "Negative"]
    df_pos = df[df["sentiment"] == "Positive"]
    df_neutral = df[df["sentiment"] == "Neutral"]


    # Example usage in Streamlit app
    st.subheader("Wordclouds with Sentiment Selection")
    
    selected_sentiment = st.selectbox("Select Semtiment", ["Positive", "Negative", "Neutral"])
    f"Word Cloud for {selected_sentiment} Reviews"

    if selected_sentiment == "Positive":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction/Wordcloud_pos.png"
        st.image(image_path)

    if selected_sentiment == "Negative":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction/Wordcloud_neg.png"
        st.image(image_path)

    if selected_sentiment == "Neutral":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction/Wordcloud_neutral.png"
        st.image(image_path)

    col1, col2 = st.columns([1, 1])
    with col1:
        # Display Word Clouds for Samsung and Apple
        st.subheader("Wordclouds with Brand and Sentiment Selection")
        '''
        In the graph presented below, we have aggregated data on the 10 most reviewed brands. To provide insights for weak points in the supply chain,
        we narrowed our focus to the three most reviewed brands: Samsung, Apple, and Blu. Subsequently, we generated Word Clouds
        tailored to each of these brands, categorizing reviews into positive, neutral, and negative sentiments.
        This detailed analysis offers a nuanced understanding of the areas where Samsung, Apple, and Blu can enhance their respective supply chains.
        '''
    with col2:
        # Display the image
        image_path = "/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction/Top10_mostReviewed.png"
        st.image(image_path, width = 400)
    
    selected_brand = st.selectbox("Select Brand", ["Samsung", "Apple", "Blu"])
    # Display Word Cloud for Selected Brand and Sentiment
    f"Word Cloud for {selected_brand} Reviews"

     # Display selected DataFrame based on brand and sentiment
    if selected_brand == "Samsung":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/SamsungWordcloud.png"
        st.image(image_path)
    elif selected_brand == "Apple":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/DataScienceBootcamp/Project/AmazonReviewsSentimentAnalysisandPrediction/Wordcloud_apple.png"
        st.image(image_path)
    elif selected_brand == "Blu":
        #Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/Wordcloud_blu.png"
        st.image(image_path)
if page == pages[3] :
    st.write("### Word2Vec Visualization for Insights into the Supply Chain")

    st.write("Word2vec is a technique in NLP to obtain vector representation of word. These vectors capture the information"
             "about the meaning of the word and their usage in the context.",
             'Using Principal Component Analysis (PCA) we visualize the relationships between the word vectors in a Word2Vec model:',
             'We selected pairs of words from our Reviews, one aspect being words like "screen," "camera," "return," "quality," "problem,"',
             '"service," "seller," and "battery" and the other being positively or negatively associated words like "great" or "worst", etc.',
             "The examples below for first all Reviews and then brand-specific Reviews will help us to understand the relationship",
             "between these words in the large corpora.")

    #Display the Word2Vec for all the review
    image_path="/Users/paulabeck/Desktop/pngs/Word2Vec_all.png"
    st.image(image_path, width = 700)

    selected_brand = st.selectbox("Select Brand", ["Samsung", "Apple", "Blu"])
    # Display Word2Vec for Selected Brand and Sentiment
    f"Word2Vec for {selected_brand} Reviews"

    # Display selected Word2Vec based on brand 
    if selected_brand == "Samsung":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/Word2Vec_Samsung.png"
        st.image(image_path, width = 700)
    elif selected_brand == "Apple":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/Word2Vec_Apple.png"
        st.image(image_path, width = 700)
    elif selected_brand == "Blu":
        #Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/Word2Vec_Blu.png"
        st.image(image_path, width = 700)
    
    st.subheader("Words in context")
    selected_words = st.selectbox("Select a Word", ["plastic","package", "satisfied", "back"])
    
    # 
    f"Associated words and context for the word'{selected_words}' for each of the 3 brands"
    if selected_words == "plastic":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/Words_plastic.png"
        st.image(image_path)
    elif selected_words == "package":
        # Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/Words_package.png"
        st.image(image_path)
    elif selected_words == "satisfied":
        #Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/Words_satisfied.png"
        st.image(image_path)
    elif selected_words == "back":
        #Display the image
        image_path = "/Users/paulabeck/Desktop/pngs/Words_back.png"
        st.image(image_path)
        
if page == pages[4] :
    st.write("### Machine and Deep Learning Models")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Multiclass Sentiment Classification", "Binary Sentiment Classification","Multiclass CNN", "Binary CNN","Regression Model"])

    with tab1:
        st.write("### Modelling with Multiclass Sentiment Classification")
        st.write("X contains the text data (reviews), and y contains the corresponding sentiment labels (positive, negative, or neutral).",
        "The data is split into a training and test set using the train_test_split function from scikit-learn.",
        "It allocates 75%  the data to training and 25% to testing. We use the CountVectorizer to convert the text data into numerical vectors.",
        "It represents each single review as a bag-of-words, counting the occurrences of each word")
        
        import pickle
        import joblib
        from sklearn.metrics import confusion_matrix
        import streamlit as st

        options = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting', 'XGBoost']
        option = st.selectbox('Choose a model', options)
        st.write('The chosen model is:', option)

        if option in ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting']:
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
                
        elif option == 'XGBoost':
            display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix'))
            if display == 'Accuracy':
                st.write(f'Accuracy for {option}:')
                # Display accuracy
                image_path = "/Users/paulabeck/Desktop/pngs/acc_xgboost.png"
                st.image(image_path, width=200)
            elif display == 'Confusion matrix':
                # Display confusion matrix 
                image_path = "/Users/paulabeck/Desktop/pngs/cf_xgboost.png"
                st.image(image_path, width=400)

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
        
        options = ['Random Oversampling', 'Random Undersampling', 'SMOTE']
        option = st.selectbox('Choose a model', options)
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
                st.image(classification_report_path, width = 800)
            else:
                st.warning(f"Image not found: {classification_report_path}")
        elif display == 'Confusion matrix':
            if os.path.exists(confusion_matrix_path):
                st.image(confusion_matrix_path, width = 500)
            else:
                st.warning(f"Image not found: {confusion_matrix_path}")
                
        
        img_path= ("/Users/paulabeck/Desktop/pngs/30mostimportantfeatures.png")
        st.image(img_path, width = 600)
        
    with tab2:
        st.write("### Modelling with Binary Classification Problem")
        st.write("The aim is to define whether Review Comment is either Negative or Positive.")
        
        options = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'XGBoost']
        option = st.selectbox('Choose a model', options)
        st.write('The chosen model is:', option)
        
        parameter_binary = st.radio('What do you want to show?', ("Classification Report", 'Accuracy', 'Confusion Matrix'))
        st.write('Model Evaluation by:', parameter_binary)
    
        # Display the chosen information
        if option == "Random Forest":
            if parameter_binary == "Classification Report":
                #Display report 
                image_path = "/Users/paulabeck/Desktop/pngs/class_rf_binary.png"
                st.image(image_path, width = 800)
            elif parameter_binary == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/paulabeck/Desktop/pngs/cf_rf_binary.png"
                st.image(image_path, width=400)
            elif parameter_binary == "Accuracy":
                #Display accuracy score
                image_path = "/Users/paulabeck/Desktop/pngs/acc_rf_binary.png"
                st.image(image_path, width = 200)
            
        if option == "Decision Tree":
            if parameter_binary == "Classification Report":
                #Display report 
                image_path = "/Users/paulabeck/Desktop/pngs/class_dt_binary.png"
                st.image(image_path, width = 800)
            elif parameter_binary == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/paulabeck/Desktop/pngs/cf_dt_binary.png"
                st.image(image_path, width=400)
            elif parameter_binary == "Accuracy":
                #Display accuracy score
                image_path = "/Users/paulabeck/Desktop/pngs/acc_dt_binary.png"
                st.image(image_path, width = 200)
        
        if option == "Logistic Regression":
            if parameter_binary == "Classification Report":
                #Display report 
                image_path = "/Users/paulabeck/Desktop/pngs/class_lr_binary.png"
                st.image(image_path, width = 800)
            elif parameter_binary == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/paulabeck/Desktop/pngs/cf_lr_binary.png"
                st.image(image_path, width=400)
            elif parameter_binary == "Accuracy":
                #Display accuracy score
                image_path = "/Users/paulabeck/Desktop/pngs/acc_lr_binary.png"
                st.image(image_path, width = 200)
        
        if option == "XGBoost":
            if parameter_binary == "Classification Report":
                #Display report 
                image_path = "/Users/paulabeck/Desktop/pngs/class_xgboost_binary.png"
                st.image(image_path, width = 800)
            elif parameter_binary == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/paulabeck/Desktop/pngs/cf_xgboost_binary.png"
                st.image(image_path, width=400)
            elif parameter_binary == "Accuracy":
                #Display accuracy score
                image_path = "/Users/paulabeck/Desktop/pngs/acc_xgboost_binary.png"
                st.image(image_path, width = 200) 
        
    with tab3:
        st.subheader("Modelling with Neural Networks for Multiclass Sentiment Classification")
        st.write("For building the CNN we used the Embedding matrix from Word2Vec.")
    
        image_path = "/Users/paulabeck/Desktop/pngs/model_sequential.png"
        st.image(image_path, width = 500)
        
        parameter_tab4 = st.radio('What do you want to show?', ("Classification Report", 'Training/Validation Accuracy/Loss', 'Confusion Matrix'), key='tab4_parameter')
        st.write('Model Evaluation by:', parameter_tab4)
        if parameter_tab4 == "Classification Report":
            # Display report 
            image_path = "/Users/paulabeck/Desktop/pngs/class_cnn.png"
            st.image(image_path, width = 800)
        elif parameter_tab4 == "Training/Validation Accuracy/Loss":
            # Display plot
            image_path = "/Users/paulabeck/Desktop/pngs/AccuracyCNN.png"
            st.image(image_path) 
        elif parameter_tab4 == "Confusion Matrix":
            # Display Confusion Matrix
            image_path = "/Users/paulabeck/Desktop/pngs/cf_cnn.png"
            st.image(image_path, width=400)

    with tab4:
        st.subheader("Modelling with Neural Networks for Binary Sentiment Classification") 
        st.write("The aim is to use Deep Learning to distinguish between positive and negative Sentiment.")
        
        parameter_tab5 = st.radio('What do you want to show?', ("Classification Report", 'Training/Validation Accuracy/Loss', 'Confusion Matrix'), key='tab5_parameter')
        st.write('Model Evaluation by:', parameter_tab5)
        
        if parameter_tab5 == "Classification Report":
            # Display report 
            image_path = "/Users/paulabeck/Desktop/pngs/class_cnn_binary.png"
            st.image(image_path, width = 800)
        elif parameter_tab5 == "Training/Validation Accuracy/Loss":
            # Display plot
            image_path = "/Users/paulabeck/Desktop/pngs/AccuracyCNN_binary.png"
            st.image(image_path) 
        elif parameter_tab5 == "Confusion Matrix":
            # Display Confusion Matrix
            image_path = "/Users/paulabeck/Desktop/pngs/cf_cnn_binary.png"
            st.image(image_path, width=400)    
    with tab5:
        st.write("### Regression Model")
        st.write("With this regression model we aim to identify a continuous value instead of a class."
                "Meaning, we want to predict the Rating or Stars given - from 1 to 5.")
        st.write("For the prediction we used Lasso Regression:")
        
        parameter_tab3 = st.radio('What do you want to show?', ("Scores", 'Prediction vs. Real'), key='tab3_parameter')
        st.write('Model Evaluation by:', parameter_tab3)
        
        if parameter_tab3 == "Scores":
            # Display report 
            image_path = "/Users/paulabeck/Desktop/pngs/linreg_scores.png"
            st.image(image_path, width=400)
        elif parameter_tab3 == "Prediction vs. Real":
            # Display plot
            image_path = "/Users/paulabeck/Desktop/pngs/linreg_matrix.png"
            st.image(image_path, width=400)
            
if page == pages[5] :
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Handle NaN values in the 'Text' column
    df = df.dropna(subset=['Text'])
    @st.cache_data
    def preprocess_data(df):
        # Splitting into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(df.Text, df.sentiment, test_size=0.25, random_state=42)
        
        # Using CountVectorizer to transform
        vectorizer = CountVectorizer()
        X_train_transformed = vectorizer.fit_transform(X_train)
        
        return X_train_transformed, y_train, vectorizer

    @st.cache_data
    def train_model(_X_train_transformed, y_train):
        # Train the RandomForestClassifier model
        clf = RandomForestClassifier(n_jobs=-1)
        clf.fit(_X_train_transformed, y_train)
        return clf

    # Streamlit app
    st.write("### Sentiment Prediction App")

    # Preprocess the data
    X_train_transformed, y_train, vectorizer = preprocess_data(df)

    # Train the model
    clf = train_model(X_train_transformed, y_train)

    # Input for user review
    user_review = st.text_area("Enter your review here to predict the Sentiment with Random Forest:")

    if st.button("Predict Sentiment"):
        # Transform user input using the pre-trained CountVectorizer
        user_input_transformed = vectorizer.transform([user_review])

        # Make prediction using the RandomForestClassifier model
        prediction = clf.predict(user_input_transformed)

        # Display the result
        st.write("**Sentiment Prediction:**")
        st.write(prediction[0])

    st.write("### How do we evaluate our models?")
    
    '''In summary we can say that the Random Forest model performs best - for both multiclass and binary Sentiment.
    In comparison to all the other models it has the highest precision, recall, and F1-score for Negative and Positive classes. 
    '''
    '''
    Other than expected the GridSearchCV for the Random Forest Model didn't improve the accuracy.'''
    st.write("For the multiclass analysis the Neutral class performed poorly -compared to the postive and negative sentiment.",
             "This is due to the imbalanced dataset, meaning there are significantly fewer reviews of Neutral feedback (3 Stars given) compared to the other classes.",
             "Also the neutral sentiment is more challenging to detect accurately as it lacks strong emotional language.",
             "Therefor we decided to use Random Over- and Undersampling as well as SMOTE to see wether we could further improve the accuracy by balancing out the dataset.",
             "Unfortunately, that did not improve the models.")
    
if page == pages[6] :
    
    st.write("### Contribution to the project")
    
    import webbrowser
    
    col1, col2 = st.columns([1, 2]) 
    image_path = "/Users/paulabeck/Desktop/pngs/logo-datascientest.png"
    col1.image(image_path)
   
    col2.markdown("This project was carried out as part of the Data Scientist training by DataScientest.com\n"
        "in cooperation with Sarbonne University.\n"
        "Our Project Manager was Axalia Levenchaud. \n"
        "Thank you for your support! \n #### Team Members:\n")

    
    # Valeriya Solozobova
    col2.markdown("**Valeria Solozobova**\n")
    col2.write("PhD, Biochemist")
    #image_path=
    #st.image(image_path)
    linkedin_url_valeria = "https://www.linkedin.com/in/valeriya-solozobova/"
    linkedin_logo_html_valeria = '<a href="{}" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png"/></a>'.format(linkedin_url_valeria)
    col2.markdown(linkedin_logo_html_valeria, unsafe_allow_html=True)
    
    # Paula Robina Beck
    col2.markdown("**Paula Robina Beck**\n")
    col2.write("Architecture")
    linkedin_url_paula = "https://www.linkedin.com/in/paula-robina-beck-b406a3172/"
    linkedin_logo_html_paula = '<a href="{}" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png"/></a>'.format(linkedin_url_paula)
    col2.markdown(linkedin_logo_html_paula, unsafe_allow_html=True)