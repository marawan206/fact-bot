from groq import Groq
import re

api_key = "PUT YOUR KEY HERE"
client = Groq(api_key=api_key)

while True:
    user_input = input("Please enter a statement to classify as true, false, or uncertain (or type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break

    # Pass the user input to the model
    completion = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {
                "role": "user",
                "content": f"Classify this statement as 'True', 'False', or 'Uncertain': {user_input}"
            }
        ],
        temperature=0,  # Made zero for more deterministic responses
        max_tokens=10,  # Reduced max tokens since we only need a single word response
        top_p=1,
        stream=True,
        stop=None,
    )

    # Print the model's response
    response = ""
    for chunk in completion:
        response += (chunk.choices[0].delta.content or "")

    # Debug: Print the raw response from the model
    print(f"Raw response: '{response.strip()}'")

    # Attempt to extract valid responses
    extracted_response = re.search(r'\b(True|False|Uncertain)\b', response, re.IGNORECASE)

    if extracted_response:
        print(extracted_response.group(0).strip())
    else:
        print("Response is not valid. Please ensure the model outputs only 'True', 'False', or 'Uncertain'.")
