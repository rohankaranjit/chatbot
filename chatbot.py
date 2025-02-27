


import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample chatbot responses
data = {
    "hello": "Hi there! How can I assist you today?",
    "how are you": "I'm just a bot, but I'm doing great! How about you?",
    "bye": "Goodbye! Have a great day!",
    "what is your name": "I'm an AI chatbot created to assist you.",
    "default": "I'm sorry, I don't understand. Can you rephrase?"
}

def preprocess(text):
    return text.lower()

# Get chatbot response
def get_response(user_input):
    user_input = preprocess(user_input)
    responses = list(data.keys())
    responses.append(user_input)
    
    vectorizer = CountVectorizer().fit_transform(responses)
    vectors = vectorizer.toarray()
    
    similarity = cosine_similarity(vectors)
    similar_idx = similarity[-1][:-1].argmax()
    
    if similarity[-1][similar_idx] > 0.5:
        return data[responses[similar_idx]]
    else:
        return data["default"]

# Chatbot loop
def chatbot():
    print("Chatbot: Hello! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
