
from rich import print
import pandas as pd
import ollama
import time
import os


def get_model_answer_multiple_options(question, options, model, dstype: str = 'mc') -> str:
    """
    Sends a query to the model and retrieves the response.

    Args:
        client: The Ollama client instance.
        question (str): The question to be categorized.
        options (str): The options for categorization.
        existing_answer (str): The existing answer.

    Returns:
        str: The model's response.
    """

    client = ollama.Client()

    if dstype == 'tc':
        messages = [
            {
                "role": "system",
                "content": "You are given a statement along with multiple options that represent different topics. Choose the option that best categorizes the statement based on its topic. Select the single option (e.g., A, B, C, etc.) that most accurately describes the topic of the statement."
            },
            {
                "role": "user",
                "content": "Mənə yaxın filial tapmağa kömək edin Options: A) ATM, B) FEES, C) OTHER, D) CARD, E) ACCOUNT, F) TRANSFER, G) PASSWORD, H) LOAN, I) CONTACT, J) FIND"
            },
            {
                "role": "assistant",
                "content": "J"
            },
            {
                "role": "user",
                "content": f"{question} Options: {options}"
            }
        ]

    if dstype == 'mc':
        messages = [
            {
                "role": "system",
                "content": "You are an AI tasked with selecting the most accurate answer in Azerbaijani based on a given question. You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. Choose the single letter (A, B, C, etc.) that best answers the question. Do not include any additional text."
            },
            {
                "role": "user",
                "content": "Bank krediti haqqında deyilənlərdən hansı yalnışdır? Options: A) Bağlanmış müqaviləyə uyğun olaraq qaytarılmalıdır, B) Müddətin uzaldılması şərtinin müqavilədə olması zəruridir, C) Təminatsız verilə bilər, D) Pul vəsaitinin verilməsinə dair qarantiya və zəmanət öhdəlikləri kredit anlayışına aid deyil"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": f"{question} Options: {options}"
            }
        ]
    
    else:
        raise("Invalid dstype")
    
    answer = ''
    try:
        stream = client.chat(
            model=model,  
            messages=messages,
            stream=True
        )
    except Exception as e:
        print(f"Error during streaming: {e}")
        return "Error"

    try:
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                answer += chunk['message']['content']
            else:
                print(f"Unexpected chunk format: {chunk}")
    except Exception as e:
        print(f"Error processing stream: {e}")
        return "Error"
    
    return answer


def compare_answers(actual_answer: str, predicted_answer: str) -> int:
    """
    Compare the actual answer with the predicted answer.
    
    Parameters:
    - actual_answer (str): The correct answer.
    - predicted_answer (str): The answer predicted by the model.
    
    Returns:
    - int: 1 if the answers match, otherwise 0.
    """
    return 1 if actual_answer.lower() == predicted_answer.lower() else 0