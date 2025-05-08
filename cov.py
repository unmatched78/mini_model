import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load the saved model
model = tf.keras.models.load_model('transformer_model.keras')

# Define your tokenizer (it should match what you used during training)
tokenizer = Tokenizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Function to generate a response from the model
def generate_response(input_text, max_sequence_length):
    # Preprocess input text
    processed_input = preprocess_text(input_text)
    
    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([processed_input])
    padded_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post', value=0)

    # Get the model's prediction (output is a sequence of token probabilities)
    predictions = model.predict(padded_sequence)

    # Convert the predicted indices to words
    predicted_indices = np.argmax(predictions, axis=-1)
    predicted_words = tokenizer.sequences_to_texts(predicted_indices)

    # Join the predicted words into a response
    response = ' '.join(predicted_words)
    return response

# Function to interact with the model
def start_conversation(max_sequence_length):
    print("Chatbot: Hi! How can I help you today?")
    print("Type 'exit' to end the conversation.")

    while True:
        # Get user input
        user_input = input("\nYou: ")

        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a nice day!")
            break

        # Generate a response based on user input
        response = generate_response(user_input, max_sequence_length)

        # Display the response
        print(f"Chatbot: {response}")

# Start the conversation
start_conversation(max_sequence_length=71)  # You may adjust max_sequence_length as needed
