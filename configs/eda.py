import json

# Load comparative questions
with open('data/raw/comparative_all.json', 'r') as f:
    questions = json.load(f)

print(f"Total question pairs: {len(questions)}")
print(f"\nFirst question:")
print(f"A: {questions[0]['question_a']}")
print(f"B: {questions[0]['question_b']}")
print(f"Correct: {questions[0]['correct_answer_a']}, {questions[0]['correct_answer_b']}")

# Load synthetic examples
with open('data/synthetic/unfaithful_examples.json', 'r') as f:
    synthetic = json.load(f)

print(f"\nTotal synthetic examples: {len(synthetic)}")
print(f"\nFirst example:")
ex = synthetic[0]
print(f"Question: {ex['question']}")
print(f"Faithful: {ex['faithful_reasoning']}")
print(f"Unfaithful: {ex['unfaithful_reasoning']}")
print(f"Type: {ex['unfaithfulness_type']}")


with open('data/processed/train.json', 'r') as f:
    train_data = json.load(f)

# This combines comparative questions + synthetic examples
print(f"Training examples: {len(train_data)}")