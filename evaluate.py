from rouge_score import rouge_scorer
from tensorflow.keras.models import load_model
from preprocessing import preprocess_texts, load_and_split_data

def evaluate_model(file_path, model_path):
    _, _, test_data = load_and_split_data(file_path)
    model = load_model(model_path)

    test_sequences, tokenizer, max_length = preprocess_texts(test_data)

    predictions = model.predict(test_sequences)
    predicted_texts = tokenizer.sequences_to_texts(predictions)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    for reference, prediction in zip(test_data['summary'], predicted_texts):
        print(scorer.score(reference, prediction))
