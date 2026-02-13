from groq import Groq

def ask_llm(context, question, api_key, model):
    """
    Ask a question to the Groq LLM using the provided context.
    Returns the answer as a string.

    Parameters:
        context (str): The context to provide to the LLM.
        question (str): The question to ask.
        api_key (str): Your Groq API key.
        model (str): The Groq model to use.

    Returns:
        str: The model's answer.
    """
    # Initialize the Groq client (proxies removed)
    client = Groq(api_key=api_key)

    # Build the prompt
    prompt = f"""
Answer only using the context below.
If the answer is not present, respond with: "Not enough information in the provided context."

Context:
{context}

Question: {question}

Answer:
"""

    # Call the Groq chat completion API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Return the text response
    return response.choices[0].message.content.strip()
