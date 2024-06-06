# DataScientest_Dec23_Supply_Chain
Project: Optimization of Supply chain with the help of customers feedback

This repository contains the jupiter notebooks for our project Supply Chain, developed during our Data Scientist training at DataScientest. It also containes teh code for Streamlit presentation of our project as well as the all the neccessary files for it.

The goal of this project is to get the insights form customers review in order to improve supply chain and create the models to predicts the positivity/negativity of comments using machine or deep learning approaches

For this project we used the Kaggle database on reviews for mobile phones from Amazon:

This project was developed by the following team :

Valeria Solozobova (GitHub / LinkedIn
Paula Robina Beck (GitHub / LinkedIn
You can browse and run the notebooks: 1.Supply_chain_clean_dataset_05_06_2024 (cleaning of the dataset to improve its quality and explorative data analysis (EDA)

2.Supply_chain_project_text_preprocessing_word_clouds:04_06_2024 (the texts were prepared to be analyized, for example , the emoji were replaced,
the digits and special characters were removed, some spelling mistakes were corrected, all the letters were brought to lower case, 
stop_words were removed. Afterwards, you can see the dataset visualizations with WordClouds). 
3...(

4...

You will need to install the dependencies (in a dedicated environment) :

pip install -r requirements.txt
Streamlit app

Add explanations on how to use the app.

To run the app (be careful with the paths of the files in the app):

conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
The app should then be available at localhost:8501.
