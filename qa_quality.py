import pandas as pd
import ollama
import logging
import time
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import Levenshtein
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODELS = {
    'LLama3.1-aze-lora-fine-tuned-v1': 'llama3.1_az',
    'llama3.1': 'llama3.1',
}

MODEL_LLAMA_3_1_405B = "meta/llama-3.1-405b-instruct"
NUM_RETRIES = 3

client = ollama.Client()

def create_combined_prompt(question: str) -> str:
    return f"""
        You are an answer generator AI in Azerbaijani based on given questions. Your task is to create answers:

        **Example:**

        **Question in Azerbaijani:** Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?  
        **Generated answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın böyük miqyasda təhlili ilə məşğul olur, mikroiqtisadiyyat isə kiçik miqyasda, yəni fərdi bazarlarda və şirkətlərdə baş verən prosesləri öyrənir.

        **Your Task:**

        **Question in Azerbaijani:** {question}

        Provide a clear and accurate answer in Azerbaijani and include your answer in 1-2 sentences.
    """

def get_answer_from_local_ollama(model: str, question: str) -> str:
    prompt = create_combined_prompt(question)
    
    stream = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True)
    
    answer = ''
    try:
        for chunk in stream:
            answer += chunk['message']['content']
    except Exception as e:
        logging.error(f"Request to local Ollama failed: {e}")
    
    return answer.strip() if answer else "Error"

def generate_answers(input_file: str, output_file: str):
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        logging.info("Successfully read the Excel file.")
    except Exception as e:
        logging.error(f"Failed to read Excel file: {e}")
        return

    if 'Sual' not in df.columns:
        logging.error("Input file must contain a 'Sual' column.")
        return

    for model in MODELS.keys():
        answer_column_name = f'{model}_answer'
        if answer_column_name not in df.columns:
            df[answer_column_name] = ''

    for index, row in df.iterrows():
        question = row['Sual']

        for model in MODELS.keys():
            answer_column = f'{model}_answer'

            if pd.isna(row[answer_column]) or row[answer_column] == '':
                predicted_answer = get_answer_from_local_ollama(MODELS[model], question)
                df.at[index, answer_column] = predicted_answer

        df.to_excel(output_file, index=False, engine='openpyxl')
        logging.info(f"Processed row {index + 1}/{len(df)}")

    df.to_excel(output_file, index=False, engine='openpyxl')
    logging.info(f"Answer generation completed and results saved to {output_file}.")

def get_evaluation_score(question: str, actual_answer: str, predicted_answer: str) -> str:
    prompt = f"""
        Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted
        answer matches the actual answer based on the asked question. Provide the score only, without any additional text.

        0-10: No answer or completely incorrect
        11-30: Significant errors or missing key information
        31-50: Some errors or incomplete information, but recognizable effort
        51-70: Mostly accurate with minor errors or omissions
        71-90: Very close to the actual answer with only minor discrepancies
        91-100: Accurate or nearly perfect match

        **Question:** {question}
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
                return content.strip() if content else "Error"
        except Exception as e:
            logging.error(f"Request failed: {e}")
            if attempt < NUM_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return "Error"

    return "Error"

def calculate_bleu_score(actual_answer: str, predicted_answer: str) -> float:
    reference = actual_answer.split()
    candidate = predicted_answer.split()
    return sentence_bleu([reference], candidate)

def calculate_rouge_score(actual_answer: str, predicted_answer: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(actual_answer, predicted_answer)

def calculate_levenshtein_distance(actual_answer: str, predicted_answer: str) -> int:
    return Levenshtein.distance(actual_answer, predicted_answer)

def evaluate_answers(input_file: str, output_file: str):
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        logging.info("Successfully read the Excel file.")
    except Exception as e:
        logging.error(f"Failed to read Excel file: {e}")
        return

    if 'Sual' not in df.columns or 'Cavab' not in df.columns:
        logging.error("Input file must contain 'Sual' and 'Cavab' columns.")
        return

    for model in MODELS.keys():
        score_column_name = f'{model}_eval_score'
        bleu_column_name = f'{model}_bleu'
        rouge_column_name = f'{model}_rouge'
        levenshtein_column_name = f'{model}_levenshtein'

        if score_column_name not in df.columns:
            df[score_column_name] = 0
        if bleu_column_name not in df.columns:
            df[bleu_column_name] = 0.0
        if rouge_column_name not in df.columns:
            df[rouge_column_name] = 0.0
        if levenshtein_column_name not in df.columns:
            df[levenshtein_column_name] = 0

    for index, row in df.iterrows():
        question = row['Sual']
        actual_answer = row['Cavab']

        for model in MODELS.keys():
            predicted_answer = row[f'{model}_answer']
            eval_score = get_evaluation_score(question, actual_answer, predicted_answer)
            df.at[index, f'{model}_eval_score'] = eval_score
            
            bleu_score = calculate_bleu_score(actual_answer, predicted_answer)
            df.at[index, f'{model}_bleu'] = bleu_score

            rouge_score = calculate_rouge_score(actual_answer, predicted_answer)
            df.at[index, f'{model}_rouge'] = rouge_score['rouge1'].fmeasure

            levenshtein_score = calculate_levenshtein_distance(actual_answer, predicted_answer)
            df.at[index, f'{model}_levenshtein'] = levenshtein_score

        df.to_excel(output_file, index=False, engine='openpyxl')
        logging.info(f"Evaluated row {index + 1}/{len(df)}")

    df.to_excel(output_file, index=False, engine='openpyxl')
    logging.info(f"Evaluation completed and results saved to {output_file}.")
