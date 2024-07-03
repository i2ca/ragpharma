import os
import sys
import json
import yaml
import string

from tqdm import tqdm
from models import Llama3, Rag, Mistral, Gemma, Aya23, Phi
from sklearn.metrics import f1_score, recall_score


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
    
    if config['rag']:
        rag = Rag()

    with open(config['path_file'], 'r') as file:
        for line in file:
            list_json.append(json.loads(line))

    list_answer_model = []
    list_gold_answer = []
    list_rag_id = []
    list_gold_id = []

    errou_junto = 0
    errou_separado = 0

    for data in tqdm(list_json):

        if config['verbose']:
            print(data['query'])

        question_prompt, letter_choices = create_multiple_choice_prompt(data['query'], data['choices'])

        if config['rag']:
            question_prompt, id_rag = add_rag_context(question_prompt, data['query'], rag)
        question_prompt = "[INST]Answer the question: [/INST] " + question_prompt + "\nAnswer: "
        # question_prompt = "[INST]Select the right option that answer the question between [][/INST] " + question_prompt + "\nAnswer: "
        # print(model.count_tokens(question_prompt))
        if config['verbose']:
            print(f"Token count: {model.count_tokens(question_prompt)}")
        # print(question_prompt)
        answer_model = multiple_choice_perplexity(model,question_prompt, data['choices'], config['verbose'])

        if config['verbose']:
            print(f"Gold {data['gold']} \nModel: {answer_model}")

        list_answer_model.append(answer_model)
        list_gold_answer.append(int(data['gold']))


        if config['rag']:
            list_rag_id.append(id_rag)
            list_gold_id.append(data['id'])
            if id_rag != data['id']:
                if answer_model != int(data['gold']):
                    errou_junto = errou_junto + 1
                else:
                    errou_separado = errou_separado + 1

    accuracy_score = accuracy(list_gold_answer, list_answer_model)
    
    f1 = f1_score(list_gold_answer, list_answer_model, average="macro")
    recall = recall_score(list_gold_answer, list_answer_model, average="macro")
    print(f"Accuracy: {accuracy_score}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")


    if config['rag']:
        print('\n')
        accuracy_rag = accuracy(list_gold_id, list_rag_id)
        print(f"Accuracy RAG: {accuracy_rag}")
        
        print(f"Errou junto: {errou_junto}")
        print(f"Errou separado: {errou_separado}")

    results = {"Model": config['model'],
               "Rag": config['rag'],
               "Accuracy": '{0:.3g}'.format(accuracy_score),
               "F1 Score": '{0:.3g}'.format(f1),
               "Recall": '{0:.3g}'.format(recall),
               "File":config['path_file']}
    if config['rag']:
        results['Accuracy RAG'] = '{0:.3g}'.format(accuracy_rag)
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
