import os
# Set your own HuggingFace token here if needed for gated models
# os.environ['HF_TOKEN'] = "your-hf-token"

from transformers import BertModel, BertTokenizer, pipeline

# Try to use a fine-tuned BERT model for fake news detection
# This model was fine-tuned on fake news classification
try:
    from transformers import AutoTokenizer, AutoModel
    print("Trying jy46604790/Fake-News-Bert-Detect...")
    tokenizer2 = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
    print("Tokenizer loaded! Vocab size:", tokenizer2.vocab_size)
    print("Model config:", tokenizer2.name_or_path)
except Exception as e:
    print(f"Error: {e}")

# Check if we can use pipeline for text classification
print("\n--- Testing text classification pipeline ---")
test_headlines = [
    "Scientists discover cure for cancer using common household spice",
    "Local cat elected mayor in historic landslide victory",
    "City council votes to increase teacher salaries by 15%",
    "BREAKING: Aliens land in central park",
]

# Try the fine-tuned model for text-only classification
try:
        classifier = pipeline(
            "text-classification",
            model="jy46604790/Fake-News-Bert-Detect",
            top_k=None,
        )
    for text in test_headlines:
        result = classifier(text)
        print(f"Headline: {text}")
        print(f"  Result: {result}")
except Exception as e:
    print(f"Pipeline error: {e}")
    print("Trying hamzab/roberta-fake-news-classification...")
    try:
        classifier = pipeline(
            "text-classification",
            model="hamzab/roberta-fake-news-classification",
            top_k=None,
        )
        for text in test_headlines:
            result = classifier(text)
            print(f"Headline: {text}")
            print(f"  Result: {result}")
    except Exception as e2:
        print(f"RoBERTa pipeline error: {e2}")
