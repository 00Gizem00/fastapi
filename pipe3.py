from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


res = classifier(
    "This is a course about Python list comprehension",
    candidate_labels=["education", "politics", "business"]
)

print(res)