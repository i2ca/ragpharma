import os
import sys
import json
import yaml
import string

from tqdm import tqdm
from models import Llama3, Rag, Mistral, Gemma, Aya23, Phi
from sklearn.metrics import f1_score, recall_score


llm_judge_prompt_1 = """
Considering the question '{question}' and the following answer '{generated_answer}', a similar alternative answer is: 
"""

llm_judge_prompt_2 = """
Considering the question '{question}' and the following answer '{generated_answer}', is '{option}' a similar alternative answer?
"""

llm_judge_prompt_3 = """
Considering the question '{question}' and the following answer '{generated_answer}', select the option with the most similar alternative answer: 
{options}
5) None of the options is similar to the text.
Option:
"""


def accuracy(ground_truth, predicted):
    """
    Calcula a acurácia de valores preditos em comparação com os valores reais.

    Args:
        ground_truth (list): Lista dos valores reais.
        predicted (list): Lista dos valores preditos.

    Returns:
        float: Acurácia do modelo.
    """
    if len(ground_truth) != len(predicted):
        raise ValueError("As listas de ground truth e preditas devem ter o mesmo comprimento.")

    correct = 0
    total = len(ground_truth)

    for i in range(total):
        if ground_truth[i] == predicted[i]:
            correct += 1

    return correct / total

def create_multiple_choice_prompt(question: str,
                                  options: list[str]):
    '''
    Function to create a multiple choice prompt.

    Args:
        question (str): The question stem.
        options (list[str]): List of options for the multiple choice question.

    Returns:
        tuple: A tuple containing the formatted question string and a list of options with associated letters.
    '''
    letters = string.ascii_uppercase[:5]
    options_with_letter = []
    full_options = ''
    
    # Create the options with the associated letter.
    for idx, option in enumerate(options):
        options_with_letter.append(f"[{letters[idx]}] {option}")
        full_options = full_options + "\n" + f"{letters[idx]}) {option}"
    # Join all options into a string.
    # full_options = "\n".join(options_with_letter)

    # Create the question with the options.
    # question = question + "\n" + full_options

    return question, options_with_letter


def multiple_choice_perplexity(model,
                               question_prompt: str,
                               options: list[str],
                               verbosity: bool):
    '''
    Function to calculate the perplexity in a multiple choice question.

    Args:
        model (Llama3): The language model used for calculating perplexity.
        question (str): The question for which perplexity is calculated.
        options (list[str]): List of options for the multiple choice question.
        verbosity (bool): Whether to print information during the calculation.

    Returns:
        int: Index of the option with the lowest perplexity.
    '''
    list_ppl = []

    for option in options:
        ppl = model.perplexity(question_prompt, option)
        list_ppl.append(ppl)
        if verbosity:
            print(f"Perplexity {option}: {ppl}")
    
    min_value = min(list_ppl)
    min_index = list_ppl.index(min_value)
    return min_index

def multiple_choice_perplexity_2(model,
                               query: str,
                               answer_model: str,
                               options: list[str],
                               verbosity: bool):

    list_ppl = []

    for option in options:
        prompt_2 = llm_judge_prompt_2.format(question=query, generated_answer=answer_model, option=option)
        ppl = model.perplexity(prompt_2, "Yes")
        list_ppl.append(ppl)
        if verbosity:
            print(f"Perplexity {option}: {ppl}")
    
    min_value = min(list_ppl)
    min_index = list_ppl.index(min_value)
    return min_index


def multiple_choice_perplexity_3(model,
                               query: str,
                               answer_model: str,
                               options: list[str],
                               verbosity: bool):

    list_ppl = []
    all_choices = '\n'.join([f"{index}) {content}" for index, content in enumerate(options)])
    for idx, choice in enumerate(options):

        prompt_3 = llm_judge_prompt_3.format(question=query, generated_answer=answer_model, options=all_choices)
        ppl = model.perplexity(prompt_3, f"{idx}) {choice}")
        list_ppl.append(ppl)
        if verbosity:
            print(f"Perplexity {choice}: {ppl}")
    
    min_value = min(list_ppl)
    min_index = list_ppl.index(min_value)
    return min_index

def add_rag_context(question_prompt: str,
                    question: str, 
                    rag: Rag):
    context, id_rag = rag.retrieve(question)

    question_prompt = f"Context: {context} \n {question_prompt}"
    return question_prompt, id_rag 

def main(config):
    
    
    if config['model'] == 'llama':
        model = Llama3()
    elif config['model'] == 'mistral':
        model = Mistral()
    elif config['model'] == 'gemma':
        model = Gemma()
    elif config['model'] == 'aya23':
        model = Aya23()
    elif config['model'] == 'phi':
        model = Phi()
    else:
        raise Exception("Model for eval not found. Please choose a model between llama and mistral.")
    
    list_json = []
    
    with open(config['path_file'], 'r') as file:
        for line in file:
            list_json.append(json.loads(line))

    list_prompt_1 = []
    list_prompt_2 = []
    list_prompt_3 = []
    list_gold_answer = []


    for data in tqdm(list_json):

        if config['verbose']:
            print(data['query'])


        answer_model = data['answer_model']

        prompt_1 = llm_judge_prompt_1.format(question=data['query'], generated_answer=answer_model)
        perplexity_prompt_1 = multiple_choice_perplexity(model,prompt_1, data['choices'], config['verbose'])
        perplexity_prompt_2 = multiple_choice_perplexity_2(model,data['query'], answer_model,data['choices'], config['verbose'])
        perplexity_prompt_3 = multiple_choice_perplexity_3(model,data['query'], answer_model,data['choices'], config['verbose'])

        if config['verbose']:
            print(f"Gold {data['gold']} \nModel: {answer_model}")
            print(perplexity_prompt_1)
            print(perplexity_prompt_2)
            print(perplexity_prompt_3)
        list_prompt_1.append(perplexity_prompt_1)
        list_prompt_2.append(perplexity_prompt_2)
        list_prompt_3.append(perplexity_prompt_3)

        list_gold_answer.append(int(data['gold']))


    accuracy_prompt_1 = accuracy(list_gold_answer, list_prompt_1)
    accuracy_prompt_2 = accuracy(list_gold_answer, list_prompt_2)
    accuracy_prompt_3 = accuracy(list_gold_answer, list_prompt_3)

    print(f"Accuracy Prompt 1: {accuracy_prompt_1}")
    print(f"Accuracy Prompt 2: {accuracy_prompt_2}")
    print(f"Accuracy Prompt 3: {accuracy_prompt_3}")

    results = {"Model": config['model'],
               "Accuracy Prompt 1": '{0:.3g}'.format(accuracy_prompt_1),
               "Accuracy Prompt 2": '{0:.3g}'.format(accuracy_prompt_2),
               "Accuracy Prompt 3": '{0:.3g}'.format(accuracy_prompt_3),

               "File":config['path_file']}

    if not os.path.exists('metric_results'):
        os.makedirs('metric_results')

    with open(f"metric_results/{config['experiment_name']}.json", "w") as outfile: 
        json.dump(results, outfile, indent=4)    



if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("The configuration file is missing.")
    else:
        config_file = sys.argv[1]
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            main(config)
