import sys
import yaml
import json 
import random

import pandas as pd
from tqdm import tqdm
from models.llama3 import Llama3


from util.error_log import atualizar_log_erro
from util.topics import list_perguntas, list_assuntos, list_exemplos
from util.prompts import prompt_check_question, prompt_make_question, system_prompt_question, prompt_points, prompt_check_context

def make_prompt_check(context, json_question):
    list_checks = []
    for wrong_answer in json_question['wrong_choices']:
        list_checks.append(prompt_check_question.format(context=context, json_question= json_question['query'],wrong_answer=wrong_answer))
    
    return list_checks

def randomize_choices(json_input):

    randomized_data = []
    for json_obj in json_input:
       
        gold_choice = json_obj["gold_choice"]
        wrong_choices = json_obj["wrong_choices"]
        all_choices = [gold_choice] + wrong_choices
        random.shuffle(all_choices)
        json_obj["gold"] = all_choices.index(gold_choice)
        json_obj["choices"] = all_choices
        json_obj.pop("gold_choice")
        json_obj.pop("wrong_choices")
        randomized_data.append(json_obj)

    return randomized_data

def main(config):
    model = Llama3()
    df = pd.read_csv(config['csv_file'])

    list_questions = []

    # Percorrendo todas as linhas e colunas do DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):


        for idx, pergunta in enumerate(list_perguntas):
                if idx in config['index_topic']:
                    
                    last_question = ''

                
                    assuntos = [list_assuntos[idx]]
                    if idx == 5:
                        if "Adulto" in row[pergunta] or "adulto" in row[pergunta]:
                            assuntos.append("posologia em adultos")
                        if "Crianças" in row[pergunta] or "crianças" in row[pergunta]:
                            assuntos.append("posologia em crianças")
                        if "Idosos" in row[pergunta] or "idosos" in row[pergunta]:
                            assuntos.append("posologia em idosos")

                    # if idx == 8:
                    #     assuntos.append("sintomas de superdosagem")
                    for assunto in assuntos:
                        
                        if last_question:
                            last_question = "Create a question different from " + last_question

                        # principal_points = model.inference(prompt_points.format(context = ))
                        prompt = prompt_make_question.format(context = row['full_topic'],
                                                                section = row[pergunta],
                                                                assunto=assunto,
                                                                    nome_remedio = row['nome'],
                                                                    example=list_exemplos[idx],
                                                                    another_question = last_question)
                        # print(prompt)
                        answer = model.inference(prompt, system_prompt_question)
                        try:
                            question_json = json.loads(answer)
                            question_json['id'] = row['id']
                            question_json['nome'] = row['nome']
                            # question_json['context'] = row['full_topic']

                            last_question = question_json['query']
                            check_failed = False

                            for check in make_prompt_check(row['full_topic'], question_json):
                                check_result = (model.inference(check, None))
                                if "YES" in check_result:
                                    check_failed = True
                                    break

                            check_result = model.inference(prompt_check_context.format(context=row[pergunta], question=question_json['query'], answer=question_json['gold_choice']))
                            if "NO" in check_result:
                                check_failed = True

                            if check_failed:
                                continue
                            else:
                                list_questions.append(question_json)
                                # print(row[pergunta])
                                # print(question_json)
                        except Exception as error:
                            print(error)
                            atualizar_log_erro(error)
                            continue  

    # Colocando as alternativas como opções aleatórias
    list_questions = randomize_choices(list_questions)

    with open(config['questions_file'], 'w') as arquivo:
        for bloco_json in list_questions:
            json.dump(bloco_json, arquivo, ensure_ascii=False)
            arquivo.write('\n')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("The configuration file is missing.")
    else:
        config_file = sys.argv[1]
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            main(config)