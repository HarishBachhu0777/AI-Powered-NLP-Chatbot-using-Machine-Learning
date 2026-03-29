
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Custom dataset
data = {
    "questions": [
        "hello", "hi", "how are you",
        "what is ai", "what is nlp",
        "what is machine learning",
        "what is deep learning",
        "bye"
    ],
    "answers": [
        "Hello! How can I assist you today?",
        "Hi there!",
        "I am doing great!",
        "AI stands for Artificial Intelligence.",
        "NLP is Natural Language Processing.",
        "Machine learning allows systems to learn from data.",
        "Deep learning is a subset of machine learning using neural networks.",
        "Goodbye! Have a nice day."
    ]
}

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["questions"])

# Model
model = MultinomialNB()
model.fit(X, data["answers"])

# Save model (optional)
with open("data.json", "w") as f:
    json.dump(data, f)

# Chat loop
print("Advanced NLP Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ").lower()

    if user_input == "exit":
        print("Bot: Goodbye!")
        break

    user_vec = vectorizer.transform([user_input])
    response = model.predict(user_vec)

    print("Bot:", response[0])
