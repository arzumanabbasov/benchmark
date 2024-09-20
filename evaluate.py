import pandas as pd
import random
import os
from typing import List
from multiple_choice import get_model_answer_multiple_options
from rag import get_answer_from_local_ollama_context
from qa_quality import get_answer_from_local_ollama
from qa_quality import get_evaluation_score
from rag import create_combined_prompt_context
from rag import create_combined_prompt
from multiple_choice import get_evaluation_score_context
from multiple_choice import compare_answers

dataset_files = [
    "LLM_BENCH_qa.xlsx",
    "Quad_benchmark_cqa.xlsx",
    "LLM-Benchmark-reshad_tc.xlsx",
    "banking-benchmark-405b-vs-8b_mmlu_fqa.xlsx",
]

metadata = {
    "version": "1.0",
    "author": "Your Name",
    "description": "This notebook evaluates different LLM models across various benchmark types such as MMLU, FinanceQA, and Contextual QA.",
    "supported_models": ["llama3.1", "llama3.1_az"],
    "benchmark_types": {
        "": "Handles questions with simple Q&A format",
        "": "Handles questions with options where one is correct",
        "": "Handles questions with context and answers",
        "": "Handles questions with topic-based options where one is correct"
    },
    "dataset_naming_convention": {
        "_mmlu_fqa": "",
        "_cqa": "",
        "_qa": "",
        "_tc": ""
    }
}

def get_benchmark_from_filename(filename, metadata):
    for ending, benchmark_type in metadata['dataset_naming_convention'].items():
        ending = ending + '.xlsx'

        if filename.endswith(ending):
            return benchmark_type
    raise ValueError(f"Filename {filename} does not match any known benchmark type")

def handle_qa(question, actual_answer, model):
    predicted_answer = get_answer_from_local_ollama(model, question, context=None)
    score = min(max(int(float(get_evaluation_score(question, predicted_answer, actual_answer))), 0), 100)
    return score

def handle_multiple_choice(question, options, correct_option, model):
    predicted_option = get_model_answer_multiple_options(question, options=options, model=model, dstype='mc')
    print(predicted_option)
    score = compare_answers(actual_answer=correct_option, predicted_answer=predicted_option)
    return score

def handle_context_qa(question, context, actual_answer, model):
    predicted_answer = get_answer_from_local_ollama_context(model, question, context)
    score = min(max(int(float(get_evaluation_score_context(question, actual_answer, predicted_answer))), 0), 100)
    return score

def handle_topic_classification(question, topic_options, correct_topic, model):
    predicted_topic = get_model_answer_multiple_options(question, options=topic_options, model=model, dstype='tc')
    print(predicted_topic)
    score = compare_answers(actual_answer=correct_topic, predicted_answer=predicted_topic)
    return score

def run_benchmark(model_name, benchmark_type, df, results):
    scores = []
    if benchmark_type == "QA":
        for index, row in df.iterrows():
            question = row['Sual']
            actual_answer = row['Cavab']
            score = handle_qa(question, actual_answer, model_name)
            scores.append(score)
    
    elif benchmark_type == "Reshad":
        for index, row in df.iterrows():
            question = row['text']
            options = row['options']
            correct_option = row['answer']
            score = handle_topic_classification(question, options, correct_option, model_name)
            scores.append(score)

    elif benchmark_type == "ContextQA":
        for index, row in df.iterrows():
            question = row['question']
            context = row['context']
            actual_answer = row['answer']
            score = handle_context_qa(question, context, actual_answer, model_name)
            scores.append(score)

    elif benchmark_type == "Arzuman":
        for index, row in df.iterrows():
            question = row['text']
            topic_options = row['options']
            correct_topic = row['answer']
            score = handle_multiple_choice(question, topic_options, correct_topic, model_name)
            scores.append(score)

    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    if scores:
        average_score = sum(scores) / len(scores)
        results.loc[model_name, benchmark_type] = average_score

results = pd.DataFrame(columns=metadata['benchmark_types'].keys(), index=metadata['supported_models'])

for file in dataset_files:
    benchmark_type = get_benchmark_from_filename(file, metadata)
    print(f"Running {benchmark_type} benchmark for file: {file}")
    
    df = pd.read_excel(file)  
    df = df[:2]  

    for model_name in metadata['supported_models']:
        print(f"Running {benchmark_type} for model {model_name}")
        run_benchmark(model_name, benchmark_type, df, results)

print("\nAverage Scores:\n", results)

results.to_excel("benchmark_results.xlsx", engine='openpyxl')
