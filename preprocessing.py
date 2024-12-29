import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path):
    df = pd.read_parquet(file_path)
    train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data

def preprocess_texts(data, tokenizer=None, max_length=None):
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['text'])

    sequences = tokenizer.texts_to_sequences(data['text'])
    sequences = [['<sos>'] + seq + ['<eos>'] for seq in sequences]

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return sequences, tokenizer, max_length
