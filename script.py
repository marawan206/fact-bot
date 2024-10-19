from transformers import pipeline

# Load a pre-trained model for natural language understanding
classifier = pipeline('text-classification', model='facebook/bart-large-mnli')

def evaluate_fact(fact):
    # Use zero-shot classification to judge if the statement is "True" or "False"
    hypothesis_template = "This statement is {}."
    labels = ['true', 'false']
    
    # Evaluate the fact
    result = classifier(fact, candidate_labels=labels, hypothesis_template=hypothesis_template)
    
    # Choose the highest scoring label (either true or false)
    if result['labels'][0] == 'true':
        return "True"
    else:
        return "False"

if __name__ == "__main__":
    print("AI Fact Checker. Type 'exit' to stop.")
    while True:
        # Get user input
        user_input = input("Enter a fact: ")
        
        # Check if user wants to exit
        if user_input.lower() == 'exit':
            break
        
        # Evaluate the fact and give the response
        response = evaluate_fact(user_input)
        print(f"AI: {response}")
