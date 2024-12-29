from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention
from tensorflow.keras.models import Model

def create_seq2seq_model(vocab_size, max_length, embedding_dim=128, lstm_units=128):
    # Encoder
    encoder_input = Input(shape=(max_length,))
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_input)
    encoder_output, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_input = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_input)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Attention
    attention = Attention()([decoder_output, encoder_output])
    decoder_combined_context = Dense(lstm_units, activation='tanh')(attention)

    # Final output
    decoder_dense = Dense(vocab_size, activation='softmax')
    final_output = decoder_dense(decoder_combined_context)

    model = Model([encoder_input, decoder_input], final_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
