# Install OpenAI Package Version 0.28
pip install openai==0.28

# Import Packages
import openai
import pandas as pd

# Function to Generate Emails
def generate_email (prompt, model = "gpt-4"):
    response = openai.ChatCompletion.create (
        model = model,
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens = 200,
        temperature = 0.8,
    )

    return response ["choices"] [0] ["message"] ["content"]

# Function to Extract Subject Lines and Message Bodies from Generated Emails
def parse_email (response):
    lines = response.split ("\n")
    subject = None
    body = []

    for line in lines:
        if line.lower ().startswith ("subject:"):
            subject = line.split (":", 1) [1].strip ()
        
        else:
            body.append (line)
    
    return subject, " ".join (body)

# Function to Generate Spam and Ham Emails
def generate_dataset (num_spam, num_ham):
    data = []

    ## Generate Spam Emails
    spam_prompt = """
    Generate a spam email with a subject line and message body to use for testing a machine learning algorithm for spam/ham email detection.
    The subject line should start with Subject:, followed by the subject line.
    """

    for _ in range (num_spam):
        spam_response = generate_email (spam_prompt)
        subject, body = parse_email (spam_response)
        data.append ({"Subject": subject, "Body": body, "Label": "Spam"})
    
    ## Generate Ham Emails
    ham_prompt = """
    Generate a ham email with a subject line and message body to use for testing a machine learning algorithm for spam/ham email detection.
    The subject line should start with Subject:, followed by the subject line.
    """

    for _ in range (num_ham):
        ham_response = generate_email (ham_prompt)
        subject, body = parse_email (ham_response)
        data.append ({"Subject": subject, "Body": body, "Label": "Ham"})
    
    return data

# Generate the Emails Dataset
dataset = generate_dataset (num_spam = 550, num_ham = 200)

# Convert the Emails Dataset to a Data Frame
emails_df = pd.DataFrame (dataset)

# Write the Emails Dataset to a CSV File
emails_df.to_csv ("LLMEmails.csv", index = False)
