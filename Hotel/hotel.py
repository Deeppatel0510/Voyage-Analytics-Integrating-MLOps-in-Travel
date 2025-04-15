# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import svds
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'hotels.csv'

@st.cache_data
def load_data(file_path):
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

hotel_df = load_data(file_path)
if hotel_df is None:
    st.stop()

# Preprocessing
def preprocess_data(df):
    """Preprocess and encode hotel dataset."""
    # Filter users with enough interactions
    interactions_count = df.groupby('userCode')['name'].count()
    active_users = interactions_count[interactions_count >= 2].index
    filtered_df = df[df['userCode'].isin(active_users)].copy()

    # Encode 'name' to numeric values
    label_encoder = LabelEncoder()
    filtered_df['name_encoded'] = label_encoder.fit_transform(filtered_df['name'])

    return filtered_df, label_encoder

filtered_df, label_encoder = preprocess_data(hotel_df)

# Collaborative Filtering Model
class CFRecommender:
    """Collaborative Filtering Recommender."""

    def __init__(self, data, factors=8):
        self.data = data
        self.factors = factors
        self.cf_preds_df = None

    def train_model(self):
        """Train the collaborative filtering model."""

        # Filter users with at least 2 interactions
        user_interactions = self.data.groupby('userCode').size()
        valid_users = user_interactions[user_interactions >= 2].index

        if len(valid_users) < 2:
            st.error("Not enough valid users with sufficient interactions for training.")
            return

        valid_data = self.data[self.data['userCode'].isin(valid_users)]
        
        interactions = valid_data.groupby(['userCode', 'name_encoded'])['price'].sum().reset_index()

        # Fallback logic for safe splitting
        if len(interactions['userCode'].unique()) > 1:
            try:
                # Use stratified split if possible
                train_df, test_df = train_test_split(
                    interactions,
                    test_size=0.25,
                    random_state=42,
                    stratify=interactions['userCode']
                )
            except ValueError:
                # Fallback to regular split if stratification fails
                st.warning("Not enough samples for stratified splitting. Using regular split.")
                train_df, test_df = train_test_split(
                    interactions,
                    test_size=0.25,
                    random_state=42
                )
        else:
            # Regular split if only one unique user
            train_df, test_df = train_test_split(
                interactions,
                test_size=0.25,
                random_state=42
            )

        # Create pivot table
        pivot = train_df.pivot(index='userCode', columns='name_encoded', values='price').fillna(0)
        pivot_matrix = pivot.values
        user_ids = list(pivot.index)

        # Matrix factorization
        k_factors = min(self.factors, min(pivot_matrix.shape) - 1)
        U, sigma, Vt = svds(pivot_matrix, k=k_factors)
        sigma = np.diag(sigma)
        all_user_ratings = np.dot(np.dot(U, sigma), Vt)

        # Store predictions
        self.cf_preds_df = pd.DataFrame(all_user_ratings, columns=pivot.columns, index=user_ids)

    def recommend_hotels(self, user_id, topn=5):
        """Generate hotel recommendations."""
        if self.cf_preds_df is None or user_id not in self.cf_preds_df.index:
            return pd.DataFrame(), f"User ID {user_id} not found in recommendations."

        # Get predictions and sort by strength
        predictions = self.cf_preds_df.loc[user_id].sort_values(ascending=False).reset_index()
        predictions.columns = ['name_encoded', 'recStrength']

        # Retrieve hotel names
        predictions['name'] = label_encoder.inverse_transform(predictions['name_encoded'])

        # Return top N recommendations
        return predictions[['name', 'recStrength']].head(topn), None

# Train the model
recommender = CFRecommender(filtered_df)
recommender.train_model()

# Streamlit Interface
st.title("Hotel Recommendation System")
st.write("Get personalized hotel recommendations based on your booking history!")

# Dropdown for user selection
user_list = filtered_df['userCode'].unique()
selected_user = st.selectbox("Select User ID:", user_list)

# Recommendation section
if st.button("Get Recommendations"):
    recommendations, error = recommender.recommend_hotels(selected_user)

    if error:
        st.warning(error)
    elif recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.success("Recommended Hotels:")

        # Improve table formatting
        recommendations['recStrength'] = recommendations['recStrength'].round(2)  # Round scores
        st.dataframe(
            recommendations.style.format({'recStrength': '{:,.2f}'}).set_properties(
                **{'background-color': '#f0f0f0', 'color': 'black'}
            )
        )

# Visualization
st.subheader("Dataset Overview")
st.dataframe(hotel_df.head())