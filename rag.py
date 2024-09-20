from dotenv import load_dotenv
import pandas as pd
import ollama
import logging
import time
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from openai import OpenAI
import Levenshtein
import logging

load_dotenv()

BASE_URL_LLM = "https://integrate.api.nvidia.com/v1"
MODEL_LLAMA_3_1_405B = "meta/llama-3.1-405b-instruct"
MODEL_LLAMA_3_1_8B = "meta/llama-3.1-8b-instruct"
API_KEY_LLM = "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"
NUM_RETRIES = 3
BASE_SLEEP_TIME = 1

def create_combined_prompt_context(context: str, question: str) -> str:
    """
    Create the prompt for the LLM to generate an answer based on the given context and question.
    """
    return f"""
        You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.

        **Example:**

        **Context:** Azərbaycan Respublikası Cənubi Qafqazda yerləşən bir ölkədir. İqtisadi və mədəni mərkəzi Bakı şəhəridir.
        
        **Question in Azerbaijani:** Azərbaycan Respublikasının paytaxtı haradır?

        **Generated Answer in Azerbaijani:** Bakı şəhəri.

        **Your Task:**

        **Context:** {context}

        **Question in Azerbaijani:** {question}

        Provide a clear and accurate answer based on the context, and include your answer in 1-2 sentences.
    """


def get_answer_from_local_ollama_context(model: str, question: str, context: str) -> str:
    """
    Send a prompt to the local Ollama model and retrieve the answer using the ollama library.
    """
    prompt = create_combined_prompt_context(context, question)
    
    stream = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )
    
    answer = ''
    try:
        for chunk in stream:
            answer += chunk['message']['content']
    except Exception as e:
        logging.error(f"Request to local Ollama failed: {e}")
    
    return answer.strip() if answer else "Error"

def get_evaluation_score_context(question: str, actual_answer: str, predicted_answer: str) -> str:
    """
    Generate an evaluation score between 0 and 100 by comparing the actual and predicted answers.
    """
    client = OpenAI(base_url=BASE_URL_LLM, api_key=API_KEY_LLM)

    prompt = f"""
            Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted
            answer matches the actual answer based on the asked question. Provide the score only, without any additional text.

            0-10: No answer or completely incorrect
            11-30: Significant errors or missing key information
            31-50: Some errors or incomplete information, but recognizable effort
            51-70: Mostly accurate with minor errors or omissions
            71-90: Very close to the actual answer with only minor discrepancies
            91-100: Accurate or nearly perfect match

            **Example:**

            **Question that asked in Azerbaijani:** Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?  
            **Actual Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın böyük miqyasda təhlili ilə məşğul olur, mikroiqtisadiyyat isə kiçik miqyasda, yəni fərdi bazarlarda və şirkətlərdə baş verən prosesləri öyrənir.  
            **Predicted Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın ümumi aspektlərini öyrənir, mikroiqtisadiyyat isə fərdi bazarları təhlil edir.  
            **Score (0 to 100):** 65

            **Your Task:**

            **Question that asked:** {question}

            **Actual Answer:** {actual_answer}

            **Predicted Answer:** {predicted_answer}

            **Score (0 to 100):**
            """


    payload = {
        "model": MODEL_LLAMA_3_1_405B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }

    for attempt in range(NUM_RETRIES):
        try:
            completion = client.chat.completions.create(**payload)
            if completion.choices:
                content = completion.choices[0].message.content
                if content:
                    return content.strip()
                logging.error("Content in response is None.")
            else:
                logging.error(f"Unexpected response format: {completion}")
        except Exception as e:
            logging.error(f"Request failed: {e}")
            if attempt < NUM_RETRIES - 1:
                sleep_time = BASE_SLEEP_TIME * (2 ** attempt)
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                return "No score received"
    return "Error"

