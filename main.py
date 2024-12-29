from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    file_path = "data/ds1.parquet"

    # Eğitim
    print("Model Eğitimi Başlatılıyor...")
    train_model(file_path)

    # Değerlendirme
    print("Model Değerlendirmesi Yapılıyor...")
    evaluate_model(file_path, 'seq2seq_model.h5')
