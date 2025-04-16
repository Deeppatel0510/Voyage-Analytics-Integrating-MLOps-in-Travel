import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

# Load models
scaler_model = pickle.load(open("scaler.pkl", 'rb'))
pca_model = pickle.load(open("pca.pkl", 'rb'))
logistic_model = pickle.load(open("tuned_logistic_regression_model.pkl", 'rb'))

# Load the Sentence Transformer model
st_model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

# Function for prediction
def predict_gender(input_data, lr_model, pca, scaler):
    text_columns = ['name']
    df = pd.DataFrame([input_data])
    
    label_encoder = LabelEncoder()
    df['company_encoded'] = label_encoder.fit_transform(df['company'])
    
    for column in text_columns:
        df[column + '_embedding'] = df[column].apply(lambda text: st_model.encode(text))
    
    n_components = 23
    text_embeddings_pca = np.empty((len(df), n_components * len(text_columns)))
    
    for i, column in enumerate(text_columns):
        embeddings = df[column + '_embedding'].values.tolist()
        embeddings_pca = pca.transform(embeddings)
        text_embeddings_pca[:, i * n_components:(i + 1) * n_components] = embeddings_pca
    
    numerical_features = ['code', 'company_encoded', 'age']
    X_numerical = df[numerical_features].values
    X = np.hstack((text_embeddings_pca, X_numerical))
    X = scaler.transform(X)
    
    y_pred = lr_model.predict(X)
    return 'Male' if y_pred[0] == 1 else 'Female'

# Streamlit UI
st.title("Gender Classification Model")

name = st.text_input("Username", "Charlotte Johnson")
usercode = st.number_input("Usercode", min_value=0, max_value=1339, step=1)
age = st.number_input("Traveller Age", min_value=21, max_value=65, step=1)
company = st.selectbox("Company Name", ["Acme Factory", "Wonka Company", "Monsters CYA", "Umbrella LTDA", "4You"])

if st.button("Predict Gender"):
    input_data = {'code': usercode, 'company': company, 'name': name, 'age': age}
    prediction = predict_gender(input_data, logistic_model, pca_model, scaler_model)
    st.success(f"Predicted Gender: {prediction}")