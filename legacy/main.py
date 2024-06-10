

import os
import json
import yaml
import random

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai




list_prompts_check = [
    "Dado esse contexto: {}, essa pergunta: [{}], avalie se a pergunta usa informações que não estão no contexto. Responda 'ERR' se ela usar, e 'OK' se ela não usar.",
    "Dado esse contexto: {}, essa pergunta: [{}] com a resposta {}, avalie se na pergunta aparece o nome do remédio {}. Responda 'OK' se tiver, e 'ERR' se ela não tiver.",
    "Dado esse contexto: {}, essa pergunta: [{}], avalie se a resposta {} está dentro do contexto. Responda 'OK' se sim e 'ERR' se não.",
   # "Dado esse contexto: {}, essa pergunta: [{}] com a resposta {}, avalie se a pergunta se confunde ao indicar algo que não abrange a todos, exemplo: se refere a posologia em adultos mas não cita na pergunta e pode ter conflito com a posologia em adolescentes ou crianças. Responda 'OK' se sim e 'ERR' se não."
]

list_prompt_check_answer = [
  "Dado esse contexto: {}, essa pergunta: {}, a resposta {} está certa? Justifique.",
  "Dado esse contexto: {}, essa pergunta: {}, a resposta {} está certa? Responda 'OK' se tiver, e 'ERR' se ela não tiver.",
]

def make_prompt(context, nome_remedio):
  return f"""Dado esses pontos sobre o remédio {nome_remedio}:
  {context}
  Crie uma pergunta com múltipla escolha, tendo apenas e somente UMA resposta certa.
  As outras alternativas NÃO PODEM ESTAR CORRETAS.
  Confira antes de gerar a pergunta se alguma alternativa errada pode estar certa. Se poder, gere outra pergunta.
  Inclua o nome do medicamento {nome_remedio} na pergunta.
  Não generalize.
  Retorne no formato json com:
  {{"query": pergunta, "choices": alternativas, "gold":" index da resposta certa de 0 a 3 correspondente ao choices."}}.
  Exemplo:
  {{"query": "Qual das seguintes indicações é tratada pelo Secni?", "choices": ["Artrite reumatoide", "Amebíase intestinal", "Doença cardiovascular", "Hipertensão"], "gold": 1}}
  Responda sem o markdown e sem quebra de linha.""" 


def main(config):

    # Config the api key and select the model
    genai.configure(api_key=config['google_key'])
    model = genai.GenerativeModel('models/gemini-pro')

    # Read the csv
    bulario = pd.read_csv(config['data_file'])

    # Select few for test
    #bulario = bulario[:3]
    bulario = bulario.sample(n=100)

    lst_questions = []

    log_error = ''
    
    for index, row in tqdm(bulario.iterrows(), total=bulario.shape[0]):
        med_name = row['Nome']
        for i in tqdm(range(0, len(row['texto']), 3000), leave=False):
            seg = row['texto'][i:i+3000]

            #seg = row['texto']

            # Se o contexto no final for muito pequeno, ignorar ele.
            if len(seg) < 200:
                continue
            try:
                prompt_points = f"Dado esse contexto: {seg} sobre esse remédio {med_name} retorne os pontos principais dele, com a posologia, indicações, princípios ativos, interações medicamentosas e etc. Sem markdown."
                points = model.generate_content(prompt_points).text
                prompt = make_prompt(points, med_name)
                question = model.generate_content(prompt).text
                
                # print(points)

                ## Conferindo se o JSON veio no formato certo:
                question_json = json.loads(question)
                question_json['full_context'] = row['texto']
                question_json['context'] = seg
                question_json['med_name'] = med_name
                question_json['points'] = points

                if "query" not in question_json and "choices" not in question_json and "gold" not in question_json:
                    raise Exception("Erro!!!! Alguma das tags faltando!")
                if not isinstance(question_json["choices"], list):
                    raise Exception("Erro!!!! As escolhas não são uma lista!")
                
                ## Conferências e validações

                check_question = True
                wrong_answers =  question_json["choices"][:question_json["gold"]] + question_json["choices"][question_json["gold"]+1 :]

                # print(question_json["choices"])
                # print(wrong_answers)

                #  # Validando as alternativas se as erradas estão certas

                # for choice in wrong_answers:
                #     prompt_check = list_prompt_check_answer[0].format(seg, question_json["query"], choice)
                #     check = model.generate_content(prompt_check).text
                #     print(check)             

                # Validando as alternativas se as erradas estão certas

                for choice in wrong_answers:
                    prompt_check = list_prompt_check_answer[1].format(seg, question_json["query"], choice)
                    check = model.generate_content(prompt_check).text
                    if check == 'OK':
                        raise Exception(f"Mais de uma alternativa correta: {question_json['query']}")

                # Validando a pergunta
                for check_prompt in list_prompts_check:
                    complete_check_prompt = check_prompt.format(seg, question_json["query"], question_json["choices"][question_json["gold"]], med_name)
                    check = model.generate_content(complete_check_prompt).text

                    if check != 'OK':
                        raise Exception(f"Pergunta invalidada {question_json['query']} no prompt {check}")
                
                if check_question:
                    lst_questions.append(question_json)
                else:
                    raise Exception("Pergunta invalidada: ", question_json)

                
            except Exception as error:
                log_error = log_error + "\n" + str(error)
                continue

    # Salvando o arquivo
    path_file = config['questions_file']

    # Removenedo se existe
    if os.path.isfile(path_file):
        os.remove(path_file)
    
    # Criando como um JSONL
    with open(path_file, 'w') as arquivo:
        for bloco_json in lst_questions:
            json.dump(bloco_json, arquivo, ensure_ascii=False)
            arquivo.write('\n')

    # Salvando log de erro
    with open('error.log', 'w') as arquivo:
        arquivo.write(log_error)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
