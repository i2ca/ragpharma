import os
import sys
import json
import yaml
import argparse
import evaluate
import torch
from tqdm import tqdm
from models import Llama3, Rag, Mistral,  Phi
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

def compute_bleu(metric, y_true, y_pred):
    metric.add_batch(predictions=y_pred, references=y_true,)
    report = metric.compute()
    bleu = report['bleu'] 
    torch.cuda.empty_cache()
    return bleu

def compute_bertscore(metric, y_pred, y_true):
    metric.add_batch(predictions=y_pred, references=y_true)
    report = metric.compute(model_type="distilbert-base-uncased")
    torch.cuda.empty_cache()
    return report

def compute_rouge(metric, y_pred, y_true):
    report = metric.compute(predictions=y_pred, references=y_true)
    torch.cuda.empty_cache()
    return report

def escreve_lista_em_arquivo(lista, nome_arquivo):
    try:
        with open(nome_arquivo, 'w') as arquivo:
            for item in lista:
                arquivo.write(f"{item}\n")
        print(f"Arquivo '{nome_arquivo}' criado com sucesso.")
    except Exception as e:
        print(f"Ocorreu um erro ao criar o arquivo: {e}")

system_prompt = """
Always answer the question in Brazilian Portuguese, being as simple and concise as possible.
Example:
Quais efeitos colaterais podem ocorrer com a administração prolongada do Mud Oral?
Answer:
Hipertensão arterial e taquicardia.
"""

def add_rag_context(question_prompt: str,
                    question: str, 
                    rag: Rag):
    context, id_rag = rag.retrieve(question)

    question_prompt = f"Context: {context} \n {question_prompt}"
    return question_prompt, id_rag 

def avg_list(list):
    return sum(list) / len(list)
 
def multiple_choice_perplexity(answer_model: str,
                               metric,
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
    list_bert_score = []

    for option in options:
        bert_score = compute_bertscore(metric, [answer_model], [option])
        list_bert_score.append(bert_score['f1'])
        if verbosity:
            print(f"BertScore F1 {option}: {bert_score['f1'][0]:.2f}")
    
    max_value = max(list_bert_score)
    min_index = list_bert_score.index(max_value)
    return min_index

def multiple_choice_perplexity_bleu(answer_model: str,
                                metric,
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
    list_bert_score = []

    for option in options:
        blue_score = compute_bleu(metric, [answer_model], [option])
        list_bert_score.append(blue_score)
        if verbosity:
            print(f"BLUE Score {option}: {blue_score:.2f}")
    
    max_value = max(list_bert_score)

    if max_value == 0: 
        return -1
    else:
        min_index = list_bert_score.index(max_value)
        return min_index

def main(config):
    
    torch.manual_seed(42)

    list_json = []
    with open(config['path_file'], 'r') as file:
        for line in file:
            list_json.append(json.loads(line))

    bleu_metric = evaluate.load('bleu', seed=42)
    bertscore_metric = evaluate.load('bertscore', seed=42)
    rouge_metric = evaluate.load('rouge', seed=42)

    list_answer_model = []
    list_answer_gold = []

    list_option_gold = []
    list_option_predicted_bert = []
    list_option_predicted_blue = []

    for data in tqdm(list_json):

        if config['verbose']:
            print(data['query'])

        answer_gold = data['choices'][data['gold']]        
        answer_model = data['answer_model']
        list_answer_model.append(answer_model)
        list_answer_gold.append(answer_gold)



        answer_choice_bleu = multiple_choice_perplexity_bleu(answer_model, bleu_metric, data['choices'], config['verbose'])
        answer_choice_bert = multiple_choice_perplexity(answer_model, bertscore_metric, data['choices'], config['verbose'])

        # list_bleu.append(bleu)
        # list_rouge.append(rouge)
        # list_bert_score.append(bert_score)
        list_option_predicted_bert.append(answer_choice_bert)
        list_option_predicted_blue.append(answer_choice_bleu)
        list_option_gold.append(int(data['gold']))

        if config['verbose']:
            print(f"Answer Gold: {answer_gold}")
            print(f"Answer Model: {answer_model}")

    bleu = compute_bleu(bleu_metric, list_answer_gold, list_answer_model)
    bert_score = compute_bertscore(bertscore_metric, list_answer_model, list_answer_gold)
    rouge = compute_rouge(rouge_metric, list_answer_model, list_answer_gold)


    bertscore_f1 = avg_list(bert_score['f1'])
    bertscore_precision = avg_list(bert_score['precision'])
    bertscore_recall = avg_list(bert_score['recall'])
    accuracy_score_bert_score = accuracy(list_option_gold, list_option_predicted_bert)
    accuracy_score_bleu = accuracy(list_option_gold, list_option_predicted_blue)

    print(f"Bert Score (F1): {bertscore_f1}")
    print(f"Bert Score (Precision): {bertscore_precision}")
    print(f"Bert Score (Recall): {bertscore_recall}")
    print(f"Accuracy Bert Choice: {accuracy_score_bert_score}")
    print(f"Accuracy BLEU: {accuracy_score_bleu}")
    print(f"BLEU: {bleu:.2f}")
    print(f"ROUGE-1: {rouge['rouge1']}")
    print(f"ROUGE-2: {rouge['rouge2']}")
    print(f"ROUGE-L: {rouge['rougeL']}")
    results = {"Bert Score (F1)": '{0:.3g}'.format(bertscore_f1),
               "Bert Score (Recall)": '{0:.3g}'.format(bertscore_recall),
               "Bert Score (Precision)": '{0:.3g}'.format(bertscore_precision),
               "Accuracy Bert Choice": accuracy_score_bert_score,
               "Accuracy BLEU": accuracy_score_bleu,
               "BLEU": '{0:.3g}'.format(bleu),
               "ROUGE-1": rouge['rouge1'],
               "ROUGE-2": rouge['rouge2'],
               "ROUGE-L": rouge['rougeL'],
            }

    if not os.path.exists(config['path_generation']):
        os.makedirs(config['path_generation'])

    with open(f"{config['path_generation']}/{config['file_generation']}", "w") as outfile: 
        json.dump(results, outfile, indent=4)    




def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument('--config_file', type=str, help='Path to the configuration file.')
    parser.add_argument('--path_file', type=str, help='Override for path_file.')
    parser.add_argument('--verbose', type=bool, help='Override for verbose.')
    parser.add_argument('--path_generation', type=str, help='Override for path_generation.')
    parser.add_argument('--file_generation', type=str, help='Override for file_generation.')
    return parser.parse_args()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("The configuration file is missing.")
    else:
        config_file = sys.argv[1]
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            main(config)
