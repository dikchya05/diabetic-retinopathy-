import argparse
import pandas as pd
from ml.models.model import train_loop  # import from model.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-csv', type=str, required=True, help='Path to CSV with labels')
    parser.add_argument('--img-dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_csv)

    # Rename 'diagnosis' to 'label' for consistency
    if 'diagnosis' in labels_df.columns:
        labels_df = labels_df.rename(columns={'diagnosis': 'label'})

    print(f"Loaded labels CSV with columns: {labels_df.columns.tolist()}")

    best_model_path = train_loop(labels_df, args.img_dir, epochs=args.epochs)
    print(f"Training completed. Best model saved at: {best_model_path}")

if __name__ == '__main__':
    main()
