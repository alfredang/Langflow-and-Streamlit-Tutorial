import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

# Set page config first
st.set_page_config(layout="wide", page_title="Text Classification Demo")

# Rebalanced training data with stronger positive indicators
training_texts = [
    # Negative examples (25 examples)
    "This product is terrible, worst purchase ever",
    "Very disappointed, poor quality and waste of money",
    "Absolutely horrible experience, would not recommend to anyone",
    "Complete garbage, broke after one day of use",
    "Terrible customer service, they ignored all my complaints",
    "Worst movie I've ever seen, boring and pointless",
    "Food was cold and tasteless, terrible restaurant",
    "The app crashes constantly, completely unusable",
    "Overpriced and underwhelming, total disappointment",
    "Poor build quality, feels cheap and flimsy",
    "Delivery was delayed by weeks, unacceptable service",
    "The book was confusing and poorly written",
    "Hotel room was dirty and uncomfortable",
    "Software is buggy and lacks basic features",
    "Staff was rude and unprofessional",
    "Product arrived damaged and customer service was unhelpful",
    "Concert was disappointing, sound quality was awful",
    "The course content was outdated and irrelevant",
    "Battery life is terrible, dies within hours",
    "Interface is confusing and hard to navigate",
    "Waste of time and money, completely useless",
    "Horrible quality, fell apart immediately",
    "Disgusting food, made me sick",
    "Boring movie, walked out halfway through",
    "Terrible experience, never going back",
    
    # Neutral examples (15 examples - reduced)
    "Product is decent, it performs adequately but could be better",
    "Product is average, it works fine but there are better options",
    "Product is okay, it does the job but there's room for improvement",
    "It's an acceptable product, nothing special but functional",
    "The service was standard, met basic expectations",
    "Movie was watchable but not memorable",
    "Food was fine, nothing to complain about but nothing extraordinary",
    "The app works as expected, basic functionality is there",
    "Fair price for what you get, neither great nor terrible",
    "Build quality is adequate for the price point",
    "Delivery was on time and packaging was standard",
    "Software does what it promises, though interface could be better",
    "Staff was polite but not particularly helpful",
    "Concert was decent, some good moments",
    "The experience was satisfactory overall",
    
    # Positive examples (45 examples - increased with more enthusiastic language)
    "I love this product, it's amazing and works perfectly",
    "Great movie, the actors are fantastic and I like the plot very much",
    "The movie is nice and the actors are fantastic, I like the plot very much",
    "Wonderful experience, highly recommend to everyone",
    "Excellent quality, exactly what I was looking for",
    "Beautiful design and fantastic functionality",
    "I'm very satisfied with this product, reliable and excellent value",
    "Product is great! Meets all expectations and performs exceptionally well",
    "I'm impressed with this product, well-designed and great functionality",
    "I absolutely love this product! Best I've ever used, highly recommended!",
    "Product exceeded expectations, it's a game-changer for my daily life",
    "Outstanding quality and performance, couldn't be happier",
    "Excellent customer service, they went above and beyond",
    "Amazing movie, captivating story and brilliant acting",
    "Delicious food and wonderful atmosphere, will definitely return",
    "The app is intuitive and works flawlessly, love it",
    "Great value for money, highly recommend to others",
    "Superior build quality, feels premium and durable",
    "Fast and reliable delivery, perfectly packaged",
    "Fantastic book, engaging and well-written throughout",
    "Beautiful hotel with exceptional service and amenities",
    "Powerful software with all the features I need",
    "Friendly and knowledgeable staff, great experience",
    "Product quality is superb, works perfectly",
    "Incredible concert, amazing performance and sound quality",
    "Excellent course content, learned so much",
    "Battery life is impressive, lasts all day",
    "User-friendly interface, easy to navigate",
    "Phenomenal experience, exceeded all expectations",
    "Top-notch quality, worth every penny",
    "Brilliant design and flawless execution",
    "Exceptional service from start to finish",
    "Outstanding performance, highly efficient",
    "Perfect for my needs, couldn't ask for more",
    "Remarkable product, innovative and reliable",
    "Superb quality and attention to detail",
    "Love the features, very useful and well-made",
    "Great product, really happy with my purchase",
    "Nice quality, good value and fast shipping",
    "Excellent service, friendly staff and clean environment",
    "Good movie, enjoyed the story and characters",
    "Really good, would definitely buy again",
    "Impressed with the quality, works as expected",
    "Happy with this purchase, meets all my needs",
    "Good experience overall, would recommend"
]

training_labels = [
    # Negative labels (25 examples)
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    
    # Neutral labels (15 examples)
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    
    # Positive labels (45 examples)
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive"
]

# Create and train classifier with better parameters
@st.cache_resource
def load_classifier():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )),
        ('classifier', MultinomialNB(alpha=0.01))  # Lower smoothing for better positive detection
    ])
    pipeline.fit(training_texts, training_labels)
    return pipeline

clf = load_classifier()

# Calculate model performance
@st.cache_data
def get_model_performance():
    scores = cross_val_score(clf, training_texts, training_labels, cv=5, scoring='accuracy')
    return scores.mean(), scores.std()

# Streamlit app
st.title("📝 Enhanced Text Classification Demo")

# Model information
with st.expander("ℹ️ Model Information"):
    st.write(f"**Training Data:** {len(training_texts)} examples")
    st.write(f"**Categories:** Negative ({training_labels.count('negative')}), Neutral ({training_labels.count('neutral')}), Positive ({training_labels.count('positive')})")
    
    accuracy, std = get_model_performance()
    st.write(f"**Model Accuracy:** {accuracy:.3f} ± {std:.3f} (5-fold cross-validation)")
    
    st.write("**Features:**")
    st.write("- TF-IDF vectorization with unigrams and bigrams")
    st.write("- Multinomial Naive Bayes classifier")
    st.write("- Trained on diverse text examples from various domains")

# Test individual text
st.subheader("🔍 Classify Your Text")
test_text = st.text_area(
    "Enter text to classify (reviews, comments, feedback, etc.):",
    placeholder="Type or paste your text here...",
    height=100
)

if st.button("🚀 Classify Text", type="primary", key="classify_button"):
    if test_text.strip():
        result = clf.predict([test_text])[0]
        
        # Get prediction probabilities for confidence
        probabilities = clf.predict_proba([test_text])[0]
        classes = clf.classes_
        
        # Create confidence scores
        confidence_dict = dict(zip(classes, probabilities))
        max_confidence = max(probabilities)
        
        # Display result with styling
        if result == "positive":
            st.success(f"✅ **Predicted Sentiment: {result.upper()}**")
        elif result == "negative":
            st.error(f"❌ **Predicted Sentiment: {result.upper()}**")
        else:
            st.info(f"➖ **Predicted Sentiment: {result.upper()}**")
        
        # Show confidence scores
        st.subheader("📊 Confidence Scores")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            neg_conf = confidence_dict.get('negative', 0)
            st.metric("Negative", f"{neg_conf:.3f}")
        
        with col2:
            neu_conf = confidence_dict.get('neutral', 0)
            st.metric("Neutral", f"{neu_conf:.3f}")
        
        with col3:
            pos_conf = confidence_dict.get('positive', 0)
            st.metric("Positive", f"{pos_conf:.3f}")
        
        # Confidence bar chart
        confidence_df = pd.DataFrame({
            'Sentiment': ['Negative', 'Neutral', 'Positive'],
            'Confidence': [
                confidence_dict.get('negative', 0),
                confidence_dict.get('neutral', 0),
                confidence_dict.get('positive', 0)
            ]
        })
        
        st.bar_chart(confidence_df.set_index('Sentiment'))
        
        # Confidence interpretation
        if max_confidence > 0.8:
            st.success("🎯 High confidence prediction")
        elif max_confidence > 0.6:
            st.warning("⚠️ Moderate confidence prediction")
        else:
            st.error("❓ Low confidence prediction - text may be ambiguous")
    else:
        st.warning("⚠️ Please enter some text to classify")