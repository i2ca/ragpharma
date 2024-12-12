import os
import sys
import json
import yaml
import string
import evaluate
import torch
from tqdm import tqdm
from models import Llama3, Rag, Mistral, Phi
from sklearn.metrics import f1_score, recall_score


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
 
def main(config):
    
    
    if config['model'] == 'llama':
        model = Llama3()
    elif config['model'] == 'mistral':
        model = Mistral()
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

    for data in tqdm(list_json):



        question_prompt = data['query']

        if config['rag']:
            question_prompt, id_rag = add_rag_context(question_prompt, data['query'], rag)
        question_prompt = "[INST]Answer the question: [/INST] " + question_prompt + "\nAnswer: "
        answer_model = model.inference(system_prompt + question_prompt)
        data['answer_model'] = answer_model
        if config['verbose']:
            print(data['query'])
            print(answer_model)

    if config['rag']:
        path_rag = 'results/generation/rag'
        if not os.path.exists(path_rag):
            os.makedirs(path_rag)
        with open(f"{path_rag}/{config['experiment_name']}.jsonl", "w", encoding='utf-8') as outfile: 
            for bloco_json in list_json:
                json.dump(bloco_json, outfile, ensure_ascii=False)
                outfile.write('\n')

    else:
        path = 'results/generation/standalone'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/{config['experiment_name']}.jsonl", "w", encoding='utf-8') as outfile: 
            for bloco_json in list_json:
                json.dump(bloco_json, outfile, ensure_ascii=False)
                outfile.write('\n')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("The configuration file is missing.")
    else:
        config_file = sys.argv[1]
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            main(config)

    # config = {
    #     "model":"llama",
    #     "rag":True,
    #     "questions_file":"results/generation/bula_right_generation_llama_biased.jsonl",
    #     "path_file":"results/bula_right.jsonl",
    #     "verbose":False
    #     }

        
    # main(config)
