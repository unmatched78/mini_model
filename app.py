# import pandas as pd
# import re

# # Load your dataset from a .txt file
# with open('text_data.txt', 'r', encoding='utf-8') as file:
#     data = file.readlines()  # Read all lines in the file

# # Convert the list of lines into a DataFrame
# data = pd.DataFrame(data, columns=['text'])

# # Display the first few entries
# print(data.head())

# # Define the preprocessing function
# def preprocess_text(text):
#     text = text.lower()  # Convert text to lowercase
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
#     return text

# # Apply the preprocessing function to the text column
# data['text'] = data['text'].apply(preprocess_text)

# # Optionally, display the processed text
# print(data.head())
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np

# # Initialize and fit the tokenizer
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(data['text'].tolist())
# total_words = len(tokenizer.word_index) + 1  # Plus one for padding

# # Tokenize the input
# input_sequences = []
# mask_token = total_words  # Assign a token ID for the mask token
# for line in data['text']:
#     token_list = tokenizer.texts_to_sequences([line])[0]
#     for i in range(len(token_list)):
#         # Randomly mask tokens
#         if np.random.rand() < 0.15:  # 15% chance to mask a word
#             original_token = token_list[i]
#             token_list[i] = mask_token  # Masking the word
#             input_sequences.append(token_list)  # Save the modified sequence
#             token_list[i] = original_token  # Restore the original token for next iterations

# # Pad sequences to the same length
# max_sequence_length = max(len(x) for x in input_sequences)
# input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# # Create features and labels
# X = input_sequences
# y = np.array([([token if token != mask_token else 0 for token in seq]) for seq in input_sequences])

# # Define the model
# import tensorflow as tf

# class TransformerBlock(tf.keras.layers.Layer):
#     def _init_(self, d_model, num_heads, dff, rate=0.1):
#         super(TransformerBlock, self)._init_()
#         self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#         self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(d_model)
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)

#     def call(self, x, training):
#         attn_output = self.attention(x, x)
#         out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        
#         ffn_output = self.dense2(self.dense1(out1))
#         return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

# class PositionalEncoding(tf.keras.layers.Layer):
#     def _init_(self, position, d_model):
#         super(PositionalEncoding, self)._init_()
#         self.pe = self.positional_encoding(position, d_model)

#     def positional_encoding(self, position, d_model):
#         # Creating the positional encodings
#         angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
#                                       np.arange(d_model)[np.newaxis, :],
#                                       d_model)

#         # Applying sine to even indices and cosine to odd indices
#         angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#         angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

#         # Convert to tensor
#         return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

#     def get_angles(self, position, i, d_model):
#         # Formula for positional encoding
#         angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
#         return position * angle_rates

# class SimpleTransformer(tf.keras.Model):
#     def _init_(self, vocab_size, d_model, num_heads, dff, max_position_encoding, num_layers):
#         super(SimpleTransformer, self)._init_()
#         self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
#         self.pos_encoding = PositionalEncoding(max_position_encoding, d_model)

#         self.encoder_layers = [TransformerBlock(d_model, num_heads, dff) for _ in range(num_layers)]
#         self.final_layer = tf.keras.layers.Dense(vocab_size)

#     def call(self, x, training):
#         seq_len = tf.shape(x)[1]
#         x = self.embedding(x)  # Embedding Layer
#         x += self.pos_encoding.pe[:, :seq_len, :]

#         for layer in self.encoder_layers:
#             x = layer(x, training)

#         return self.final_layer(x)

# # Instantiate the model with parameters
# num_layers = 4
# d_model = 128
# num_heads = 4
# dff = 512
# max_position_encoding = max_sequence_length

# model = SimpleTransformer(total_words, d_model, num_heads, dff, max_position_encoding, num_layers)

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# epochs = 10
# batch_size = 32
# model.fit(X, y, epochs=epochs, batch_size=batch_size)

# model.save('intensive_transformer_model.h5')
# def generate_text(start_string, num_words):
#     input_seq = tokenizer.texts_to_sequences([start_string])[0]
#     input_seq = pad_sequences([input_seq], maxlen=max_sequence_length-1, padding='pre')

#     for _ in range(num_words):
#         predicted_logits = model.predict(input_seq)
#         predicted_index = tf.argmax(predicted_logits[0]).numpy()
#         input_seq = pad_sequences([input_seq[0].tolist() + [predicted_index]], maxlen=max_sequence_length-1, padding='pre')

#     return tokenizer.sequences_to_texts([[predicted_index]])[0]

# # Example of generating text
# print(generate_text("Once upon a time", 10))


import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
try:
    with open('text_data.txt', 'r', encoding='utf-8') as file:
        data = file.readlines()
except FileNotFoundError:
    print("File not found. Please check the path and filename.")
    data = []

data = pd.DataFrame(data, columns=['text'])

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

data['text'] = data['text'].apply(preprocess_text)
data = data[data['text'].astype(bool)]

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
total_words = len(tokenizer.word_index) + 1  # Include padding token
mask_token = total_words - 1  # Mask token within valid range

# Create input sequences with masking
input_sequences = []
labels = []
for line in data['text']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(len(token_list)):
        if np.random.rand() < 0.15:  # 15% chance to mask a word
            masked_seq = token_list.copy()
            label = token_list.copy()
            masked_seq[i] = mask_token  # Replace with mask token
            label[i] = token_list[i]  # Original token
            input_sequences.append(masked_seq)
            labels.append(label)

# Padding
max_sequence_length = max(len(seq) for seq in input_sequences)
X = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post', value=0)
y = pad_sequences(labels, maxlen=max_sequence_length, padding='post', value=mask_token)

# Transformer model
import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()  # Register the custom class
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.attention(x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.dense2(self.dense1(out1))
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

@tf.keras.utils.register_keras_serializable()  # Register the custom class
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pe = self.positional_encoding(position, d_model)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

@tf.keras.utils.register_keras_serializable()  # Register the custom class
class SimpleTransformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, max_position_encoding, num_layers):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = PositionalEncoding(max_position_encoding, d_model)
        self.encoder_layers = [TransformerBlock(d_model, num_heads, dff) for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training=False):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return self.final_layer(x)


# Model parameters
num_layers = 4
d_model = 128
num_heads = 4
dff = 512

model = SimpleTransformer(total_words, d_model, num_heads, dff, max_sequence_length, num_layers)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dataset preparation
dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
TF_ENABLE_ONEDNN_OPTS=0
# Training
model.fit(dataset, epochs=10)

# Save model
model.save('transformer_model.keras')
