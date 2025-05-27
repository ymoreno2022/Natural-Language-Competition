import argparse
from model.predict import predict_emotion, get_kaggle_id

def main():
    parser = argparse.ArgumentParser(description="Emotion classifier CLI")
    parser.add_argument("--input", type=str, help="Text to classify")
    parser.add_argument("--kaggle", action="store_true", help="Show Kaggle ID")
    args = parser.parse_args()

    if args.kaggle:
        print(get_kaggle_id())
    elif args.input:
        print(predict_emotion(args.input))
    else:
        print("Please provide either --input or --kaggle")
