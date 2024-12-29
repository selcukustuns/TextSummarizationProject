from preprocessing import preprocess_texts, load_and_split_data

def train_model(file_path):
    train_data, val_data, _ = load_and_split_data(file_path)
    train_sequences, tokenizer, max_length = preprocess_texts(train_data)
    val_sequences, _, _ = preprocess_texts(val_data, tokenizer, max_length)

    vocab_size = len(tokenizer.word_index) + 1
    model = create_seq2seq_model(vocab_size, max_length)

    model.fit(train_sequences, train_sequences,
              validation_data=(val_sequences, val_sequences),
              batch_size=64, epochs=10)
    model.save('seq2seq_model.h5')
