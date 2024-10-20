from langchain_community.llms import ollama
from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama3.1-8b"


# Initialize the model using Ollama's llama3
model = ollama.Ollama(model="llama3")

# Fact-checker agent: takes a statement and classifies it as true or false
classifier = Agent(
    role="Fact Classifier",
    goal="Accurately classify a given statement as true or false using LLM."
)

# Define a task for classifying a fact
def classify_fact(task_input):
    statement = task_input["statement"]
    prompt = f"Is the following statement true or false?\n\n'{statement}'"
    
    # Use the model to classify
    response = model.predict(prompt)
    
    # Parse the response and determine if it's True/False
    if "true" in response.lower():
        return {"classification": "True"}
    elif "false" in response.lower():
        return {"classification": "False"}
    else:
        return {"classification": "Uncertain"}

# Define a task for generating a response based on classification
def respond_to_classification(task_input):
    classification = task_input["classification"]
    
    # Based on classification, generate a response
    if classification == "True":
        return "The statement is classified as True."
    elif classification == "False":
        return "The statement is classified as False."
    else:
        return "The statement's truthfulness is uncertain."

# Define the tasks for the process
classify_fact_task = Task(
    name="Classify Fact",
    agent=classifier,
    function=classify_fact,
    inputs={"statement": "Is the Earth flat?"}  # Example input statement
)

respond_to_classification_task = Task(
    name="Respond to Classification",
    agent=classifier,
    function=respond_to_classification,
    inputs={"classification": classify_fact_task.outputs["classification"]}
)

# Create the Crew with agents and tasks
crew = Crew(
    agents=[classifier],
    tasks=[classify_fact_task, respond_to_classification_task],
    verbose=2,
    process=Process.sequential
)

# Kickoff the process
output = crew.kickoff()
print(output)
