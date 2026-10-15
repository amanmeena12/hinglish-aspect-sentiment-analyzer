
import re
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.joblib"

# Aspect keywords
ASPECT_KEYWORDS = {
    "quality": [
        "quality", "product", "material", "build",
        "durable", "durability"
    ],
    "delivery": [
        "delivery", "deliver", "order", "shipping",
        "parcel", "arrived", "late", "pahuncha"
    ],
    "price": [
        "price", "paisa", "paise", "cost",
        "expensive", "cheap", "mehenga", "sasta", "worth"
    ],
    "customer_service": [
        "support", "service", "customer care",
        "help", "staff", "response", "refund"
    ],
    "battery": [
        "battery", "backup", "charging", "charge", "power"
    ],
    "camera": [
        "camera", "photo", "photos", "picture",
        "image", "blur"
    ],
    "packaging": [
        "packing", "packaging", "box", "package"
    ],
    "usability": [
        "app", "interface", "crash", "hang",
        "software", "easy", "user friendly"
    ],
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome",
    "best", "nice", "achha", "achhi", "accha", "acchi",
    "badhiya", "mast", "badiya", "sahi", "love",
    "perfect", "fast", "quick", "smooth", "helpful",
    "durable", "worth", "affordable", "easy"
}

NEGATIVE_WORDS = {
    "bad", "worst", "poor", "terrible", "bekar",
    "kharab", "ghatiya", "bura", "buri", "slow",
    "late", "damaged", "broken", "useless", "expensive",
    "mehenga", "problem", "issue", "crash", "hang",
    "blur", "laggy", "disappointed"
}

NEGATIONS = {
    "not", "no", "never", "nahi", "nahin", "mat",
    "bilkul nahi", "not at all"
}

ASPECT_LABELS = {
    "quality": "Quality",
    "delivery": "Delivery",
    "price": "Price",
    "customer_service": "Customer Service",
    "battery": "Battery",
    "camera": "Camera",
    "packaging": "Packaging",
    "usability": "Usability",
    "general": "General"
}


def contains_keyword(text, keyword):
    pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def detect_aspects(review):
    text = review.lower()
    detected = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(contains_keyword(text, word) for word in keywords):
            detected.append(aspect)

    return detected if detected else ["general"]


def split_sentences(review):
    # Split on common sentence boundaries and contrast words.
    parts = re.split(
        r"[.!?;,\n]+|\bbut\b|\blekin\b|\bpar\b|\bhowever\b",
        review,
        flags=re.IGNORECASE
    )
    return [part.strip() for part in parts if part.strip()]


def classify_text_sentiment(text):
    words = re.findall(r"[a-zA-Z]+|[\u0900-\u097F]+", text.lower())

    positive_score = 0
    negative_score = 0

    for i, word in enumerate(words):
        window = words[max(0, i - 2):i]
        negated = any(item in NEGATIONS for item in window)

        if word in POSITIVE_WORDS:
            if negated:
                negative_score += 1
            else:
                positive_score += 1

        elif word in NEGATIVE_WORDS:
            if negated:
                positive_score += 1
            else:
                negative_score += 1

    if positive_score > negative_score:
        return "positive"
    if negative_score > positive_score:
        return "negative"
    return "neutral"


def analyze_review(review):
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Please enter a review.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file missing. Run: python train_model.py"
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    overall_sentiment = str(model.predict([review])[0])
    probabilities = model.predict_proba([review])[0]
    confidence = float(max(probabilities))

    aspect_results = []
    sentences = split_sentences(review)

    for aspect in detect_aspects(review):
        relevant_parts = []

        for sentence in sentences:
            keywords = ASPECT_KEYWORDS.get(aspect, [])
            if aspect == "general" or any(
                contains_keyword(sentence, keyword)
                for keyword in keywords
            ):
                relevant_parts.append(sentence)

        relevant_text = " ".join(relevant_parts) or review
        aspect_sentiment = classify_text_sentiment(relevant_text)

        aspect_results.append({
            "aspect": aspect,
            "aspect_name": ASPECT_LABELS.get(aspect, aspect.title()),
            "sentiment": aspect_sentiment,
            "evidence": relevant_text
        })

    return {
        "review": review,
        "sentiment": overall_sentiment,
        "confidence": confidence,
        "aspects": [item["aspect"] for item in aspect_results],
        "aspect_results": aspect_results
    }


if __name__ == "__main__":
    print("Hinglish Aspect-Based Sentiment Analyzer")
    print("Type 'exit' to stop.")

    while True:
        review = input("\nEnter a review: ").strip()

        if review.lower() == "exit":
            break

        try:
            result = analyze_review(review)

            print("\nOverall sentiment:", result["sentiment"])
            print("Model confidence:", f'{result["confidence"]:.2%}')
            print("\nAspect-wise sentiment:")

            for item in result["aspect_results"]:
                print(
                    f'{item["aspect_name"]}: '
                    f'{item["sentiment"]} '
                    f'| Evidence: {item["evidence"]}'
                )

        except (ValueError, FileNotFoundError) as error:
            print("Error:", error)

