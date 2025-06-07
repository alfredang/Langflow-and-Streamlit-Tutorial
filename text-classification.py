import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample data
sample_data = {
    'Product': [
        'Wireless Headphones', 'Smartphone Case', 'Laptop Stand', 'Coffee Mug', 
        'Bluetooth Speaker', 'Phone Charger', 'Desk Lamp', 'Water Bottle',
        'Gaming Mouse', 'Keyboard'
    ],
    'Review': [
        "I love this product, it's amazing and the sound quality is perfect!",
        "This is terrible, worst purchase ever. Broke after one day.",
        "It's okay, nothing special but does the job fine.",
        "Great quality and fast delivery. Very satisfied with purchase.",
        "Poor service, very disappointed. Sound is awful.",
        "Excellent value for money. Works perfectly as expected.",
        "Not worth the price. Stopped working after a week.",
        "Fantastic experience, highly recommend to everyone!",
        "Average product, could be better but acceptable.",
        "Outstanding quality and service. Will buy again!"
    ]
}

# Create training data for classifier
training_texts = [
    "I love this product, it's amazing!",
    "This is terrible, worst purchase ever",
    "It's okay, nothing special",
    "Great quality and fast delivery",
    "Poor service, very disappointed",
    "Excellent value for money",
    "Not worth the price",
    "Fantastic experience, highly recommend",
    "Average product, could be better",
    "Outstanding quality and service"
]

training_labels = [
    "positive", "negative", "neutral", "positive", "negative",
    "positive", "negative", "positive", "neutral", "positive"
]

# Create and train classifier
@st.cache_resource
def load_classifier():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(training_texts, training_labels)
    return pipeline

clf = load_classifier()

# Streamlit app
st.set_page_config(layout="wide", page_title="Text Classification Demo")
st.title("📝 Simple Text Classification Demo")

# Create sample dataframe
df = pd.DataFrame(sample_data)

if st.button("Analyze Sample Data"):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("Sample Data")
        st.dataframe(df, use_container_width=True)
    
    with col2:
        # Classify the reviews
        reviews = df['Review'].tolist()
        predictions = clf.predict(reviews)
        
        # Add predictions to dataframe
        df['Sentiment'] = predictions
        
        st.info("Classification Results")
        st.dataframe(df, use_container_width=True)
        
        # Show sentiment counts
        sentiment_counts = pd.Series(predictions).value_counts()
        st.bar_chart(sentiment_counts)
        
        # Download button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results", 
            data=csv_data, 
            file_name='classified_results.csv', 
            mime='text/csv'
        )

# Test individual text
st.markdown("---")
st.subheader("Test Individual Text")
test_text = st.text_area("Enter your own text to classify:")
if st.button("Classify") and test_text:
    result = clf.predict([test_text])[0]
    st.success(f"Predicted Sentiment: **{result.upper()}**")