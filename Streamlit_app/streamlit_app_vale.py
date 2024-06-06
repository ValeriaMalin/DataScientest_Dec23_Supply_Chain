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
df = pd.read_csv('/Users/solozobovavaleria/Supply_Chain_cleaned.csv')

st.title("Supply Chain : Sentiment Analysis Project")
st.sidebar.title("Table of contents")
pages=["Our Project","Data Processing", "Wordclouds", "Word2Vec for supply chains", "Modelling", "Conclusion", "About"]
page=st.sidebar.radio("Go to", pages)


if page == pages[0] : 
    st.header("Exploratory Data Analysis and Primary Data Preprocessing")
    st.subheader(
    "With our project we wanted to gain valuable insights from cellphone reviews of  Amazon, providing a deeper understanding of the products strengths and weaknesses and to improve the Supply Chain.",
    "We were exploring the following steps:\n\n"
    )
    st.write(
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

    st.header(" Focus of Dataset")
    
    st.write(
    "To ensure a more targeted analysis, we narrowed our data down to cellphones only.",
    "This refinement allows us to extract insights directly relevant to the supply chain of mobile devices,",
    "eliminating unnecessary noise from cellphone accessories -therefor enhancing the significance of our findings.\n"
    )
    
    st.write(
    "1.The initial data preprocessing included: **removal of duplictates and rows with missing reviews,**\n\n",
    "2. **text mining on unknown brandnames** (we traced their brandname from the Product Name),\n\n"
    "3. **the reduction of inconsistent capitalization and formatting in the Brand Name column**,\n\n",
    "4. **deletion of all cellphone accessories** contained in the dataset with the focus on only the cellphones.\n\n",
    "5. **grouping the products by average rating and number of reviews** for each product.\n\n"
    )

    st.header("Distribution of Ratings")
    # Display the image
    image_path = "/Users/solozobovavaleria/Rating_Distribution.png"
    st.image(image_path)


if page == pages[1] : 
    st.header("Text Preprocessing")
    

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

    st.header("An Example")
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

    st.header("Create your own review")
    user_input = st.text_area("Enter your text:")
    text_processing(user_input)

if page == pages[2] : 
    st.header("Wordclouds")

    # Distribution of Review Lengths
    st.subheader("Distribution of Review Lengths")
    # Display the image
    image_path = "/Users/solozobovavaleria/Review Length per Rating.png"
    st.image(image_path)


    # Separate df into positive, neutral and negative data
    df_neg = df[df["sentiment"] == "Negative"]
    df_pos = df[df["sentiment"] == "Positive"]
    df_neutral = df[df["sentiment"] == "Neutral"]


    # Example usage in Streamlit app
    st.subheader("Word Clouds")
    
    
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
    st.subheader("Brand and Sentiment Selection")
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
    st.header("Word2Vec Visualization as Supply Chain Insights")

    st.write("**Word2vec is a technique in NLP to obtain vector representation of word. These vectors capture the information**"
             "**about the meaning of the word and their usage in the context**")
    st.subheader("An example of Word2Vec visualizazion on selected words from Reviews Texts")
    st.write("**Visualizaion helps to undestand the relation between words in the large corpora.**")

    #Display the Word2Vec for all the review
    image_path="/Users/solozobovavaleria/Word2Vec_all.png"
    st.image(image_path)

    selected_brand = st.selectbox("Select Brand", ["Samsung", "Apple", "Blu"])
    st.subheader("The Word2Vec visualization for Selected Brands")

    # Display selected Word2Vec based on brand 
    if selected_brand == "Samsung":
        # Display the image
        image_path = "/Users/solozobovavaleria/Word2Vec_Samsung.png"
        st.image(image_path)
    elif selected_brand == "Apple":
        # Display the image
        image_path = "/Users/solozobovavaleria/Word2Vec_Apple.png"
        st.image(image_path)
    elif selected_brand == "Blu":
        #Display the image
        image_path = "/Users/solozobovavaleria/Word2Vec_Blu.png"
        st.image(image_path)
    
    st.subheader("Words in context")
    selected_words = st.selectbox("Selected Words", ["plastic","package", "satisfied", "back"])

    # Display selected Word2Vec based on brand 
    if selected_words == "plastic":
        # Display the image
        image_path = "/Users/solozobovavaleria/Words_plastic.png"
        st.image(image_path)
    elif selected_words == "package":
        # Display the image
        image_path = "/Users/solozobovavaleria/Word_package.png"
        st.image(image_path)
    elif selected_words == "satisfied":
        #Display the image
        image_path = "/Users/solozobovavaleria/Words_satisfied.png"
        st.image(image_path)
    elif selected_words == "back":
        #Display the image
        image_path = "/Users/solozobovavaleria/supply_chain_streamlit/Words_back.png"
        st.image(image_path)

if page == pages[4] :
    st.title("Machine and Deep Learning Models")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Multiclass Sentiment Classification**", "**Binary Classification**", "**Regression Model**","**Multiclass CNN**", "**Binary CNN**"])
    options = ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'XGBoost', 'CNN_multiclass', 'CNN_binary', 'Lasso Regression']
    option = st.selectbox('Model is', options)
    parameters = ['Classification Report', 'Confusion Matrix', 'Accuracy Score','Scores', 'Prediction versus Real', 'Training/Validation Accuracy/Loss']
    parameter = st.selectbox('Evaluation technique is', parameters)
    
    with tab1:
        st.header("Modelling with Multiclass Sentiment Classification")
        st.subheader("The aim is to define whether Review Comment is Negative, Positive or Neutral.")
        

        if option == "Random Forest":
            st.write('The chosen model is:', option)
            st.write('Model Evaluation by:', parameter)    
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_rf.png"
                st.image(image_path)
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_rf.png"
                st.image(image_path)
            elif parameter == "Accuracy Score":
                #Display accuracy score
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/acc_rf.png"
                st.image(image_path)
            
        if option == "Decision Tree":
            st.write('The chosen model is:', option)
            st.write('Model Evaluation by:', parameter)
            
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_dt.png"
                st.image(image_path)
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_dt.png"
                st.image(image_path)
            elif parameter == "Accuracy Score":
                #Display accuracy score
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/acc_dt.png"
                st.image(image_path)
        
        if option == "Gradient Boosting":
            st.write('The chosen model is:', option)
            st.write('Model Evaluation by:', parameter)
            
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_gradient.png"
                st.image(image_path)
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_gradient.png"
                st.image(image_path)
            elif parameter == "Accuracy Score":
                #Display accuracy score
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/acc_gradient.png"
                st.image(image_path)
        
        if option == "XGBoost":
            st.write('The chosen model is:', option)
            st.write('Model Evaluation by:', parameter)
            
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_xgboost.png"
                st.image(image_path)
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_xgboost.png"
                st.image(image_path)
            elif parameter == "Accuracy Score":
                #Display accuracy score
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/acc_xgboost.png"
                st.image(image_path)
    
    with tab2:
        st.header("Binary Classification Problem")
        st.subheader("The aim is to define whether Review Comment is Negative or Positive ")
        
        if option == "Random Forest":
            st.write('The chosen model is:', option)
            st.write('Model Evaluation by:', parameter)
            
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_rf_binary.png"
                st.image(image_path)
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_rf_binary.png"
                st.image(image_path)
            elif parameter == "Accuracy Score":
                #Display accuracy score
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/acc_rf_binary.png"
                st.image(image_path)
            
        if option == "Decision Tree":
            st.write('The chosen model is:', option)
            st.write('Model Evaluation by:', parameter)
            
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_dt_binary.png"
                st.image(image_path)
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_dt_binary.png"
                st.image(image_path)
            elif parameter == "Accuracy Score":
                #Display accuracy score
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/acc_dt_binary.png"
                st.image(image_path)
        
        if option == "Logistic Regression":
            st.write('The chosen model is:', option)
            st.write('Model Evaluation by:', parameter)
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_lr_binary.png"
                st.image(image_path)
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_lr_binary.png"
                st.image(image_path)
            elif parameter == "Accuracy Score":
                #Display accuracy score
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/acc_lr_binary.png"
                st.image(image_path)
        
        if option == "XGBoost":
            st.write('The chosen model is:', option)
            st.write('Model Evaluation by:', parameter)
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_xgboost_binary.png"
                st.image(image_path)
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_xgboost_binary.png"
                st.image(image_path)
            elif parameter == "Accuracy Score":
                #Display accuracy score
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/acc_xgboost_binary.png"
                st.image(image_path) 
    
        
    with tab3:
        st.header("Regression problem")
        
        st.subheader("In regression model we aim to identify instead of class the continuus value")
        st.write("**In our case, the prediction of Rating or Given Star value - from 1 to 5.**")
        st.write("**We used lasso regression model:**")
        
        if option == "Lasso Regression":
            st.write('Model Evaluation by:', parameter)
            if parameter == "Scores":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/linreg_scores.png"
                st.image(image_path)
            elif parameter == "Prediction versus Real":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/linreg_matrix.png"
                st.image(image_path)        
        
    with tab4:
        st.header("Modelling with Neural Networks for Multiclass Sentiment Classification")
        st.subheader("CNN was build using Embedding matrix from Word2Vec object")

        if option == "CNN_multiclass":
            image_path = "/Users/solozobovavaleria/supply_chain_streamlit/model_sequential.png"
            st.image(image_path)
            st.write('Model Evaluation by:', parameter)
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_cnn.png"
                st.image(image_path)
            elif parameter == "Training/Validation Accuracy/Loss":
                #Display plot
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/Acc_loss_cnn.png"
                st.image(image_path) 
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_cnn.png"
                st.image(image_path)   
            
            
    with tab5:
        st.header("Modelling with Neural Networks for Binary Sentiment Classification") 
        st.subheader("The aim is to use deep learnign to distinguish betwenn postive and negative sentiments")
        if option == "CNN_binary":
            st.write('Model Evaluation by:', parameter)
            if parameter == "Classification Report":
                #Display report 
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/class_cnn_binary.png"
                st.image(image_path)
            elif parameter == "Training/Validation Accuracy/Loss":
                #Display plot
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/Acc_loss_cnn_binary.png"
                st.image(image_path) 
            elif parameter == "Confusion Matrix":
                #Display Confusion Matrix
                image_path = "/Users/solozobovavaleria/supply_chain_streamlit/cf_cnn_binary.png"
                st.image(image_path)    


        
if page == pages[5] :
    st.header("Conclusion: The best model")
    st.write ('**We tested numerous models in order to predict sentiment from reviews**')
    st.write("**Several showed accuracy scores around 90% for multiclass  problem and over 90% for binary class problem**")

    st.subheader("The Random Forest was the best for both multi- and binary classification problem")
    #here we can show feature importances
    #and the test case
    st.subheader("Feature Importances for Random Forest model")
    image_path = "/Users/solozobovavaleria/supply_chain_streamlit/30features.png"
    st.image(image_path)

    st.subheader("Test case")
    
if page == pages[6] :
    
    st.write("### Contribution to the project")
    
    import webbrowser
    
    col1, col2 = st.columns([1, 2]) 
    image_path = "/Users/paulabeck/Desktop/logo-datascientest.png"
    col1.image(image_path)
   
    col2.markdown("This project was carried out as part of the Data Scientist training by DataScientest.com\n"
        "in cooperation with Sarbonne University.\n"
        "Our Project Manager was Axalia Levanchaud. \n #### Team Members:\n")

    
    # Valeria Solozobova
    col2.markdown("Valeria Solozobova")
    st.write ("PhD, Biochemist")
    #image_path=
    st.image(image_path)
    linkedin_url_valeriya = "https://www.linkedin.com/in/valeriya-solozobova/"
    linkedin_logo_html_valeriya = '<a href="{}" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png"/></a>'.format(linkedin_url_valeriya)
    col2.markdown(linkedin_logo_html_valeriya, unsafe_allow_html=True)
    
    # Paula Robina Beck
    col2.markdown("Paula Robina Beck")
    linkedin_url_paula = "https://www.linkedin.com/in/paula-robina-beck-b406a3172/"
    linkedin_logo_html_paula = '<a href="{}" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png"/></a>'.format(linkedin_url_paula)
    col2.markdown(linkedin_logo_html_paula, unsafe_allow_html=True)