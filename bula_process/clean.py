import os
import sys
import yaml 
import PyPDF2

import pandas as pd
from tqdm import tqdm
from models.llama3 import Llama3

from util.error_log import atualizar_log_erro
from util.topics import list_perguntas, list_topics
from util.prompts import clean_profissional_prompt, clean_paciente_last_question_prompt, clean_paciente_prompt

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages) -1
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    
    # Split the text into lines and filter out empty lines
    lines = filter(lambda x: x.strip(), text.split('\n'))
    cleaned_text = '\n'.join(lines)
    
    return cleaned_text



def main(config):

    # Limpando o arquivo de log
    if os.path.isfile('log.txt'):
        os.remove('log.txt')

    pdfs_path = config['pdfs_path']
    model = Llama3()
    df = pd.DataFrame()

    if config['type'] != 'profissional' and config['type'] != 'paciente':
        raise Exception("Config type not found. Choose between 'profissional' and 'paciente'.")

    for filename in tqdm(os.listdir(pdfs_path)[:500]):

        try:

            text = extract_text_from_pdf(pdfs_path + filename)
            dict_row = {}
            if config['type'] == 'profissional':
                full_size = len(text)
                half = full_size // 2
                halfhalf = half // 2
                p1 = text[:half]
                p2 = text[halfhalf:full_size - halfhalf]
                p3 = text[half:]

                tokens_p1 = model.count_tokens(p1)

                if tokens_p1 > 7200:
                    raise Exception(f"{filename} has more than 7500 tokens in a half file.")

                list_ptr = [p1, p2, p3]
        
                nome_medicamento = model.inference(f"Given this leaflet: {p1}, what is the name of the medicine? Answer only the name.")
                dict_row['nome'] = nome_medicamento.replace('.', '')
                dict_row['id'] = filename.replace('.pdf','')
                dict_row['full_topic'] = f"Bula de {dict_row['nome']} "

                for idx, parte in enumerate(list_ptr):         

                    for topic_name in list_topics[idx]:                           
                        
                        loaded_prompt = clean_profissional_prompt.format(parte = parte, topic_name = topic_name)
                        topic = model.inference(loaded_prompt)
                        dict_row[topic_name] = topic.replace('\n', ' ')
                        dict_row['full_topic'] = dict_row['full_topic'] + f" {topic_name} {dict_row[topic_name]}"

            elif config['type'] == 'paciente':
            
                tokens = model.count_tokens(text)
                if tokens > 7500:
                    raise Exception(f"{filename} has more than 7500 tokens.")

                nome_medicamento = model.inference(f"Given this leaflet: {text}, what is the name of the medicine? Answer only the name.")
                dict_row['nome'] = nome_medicamento
                dict_row['id'] = filename.replace('.pdf','')
                dict_row['full_topic'] = f"Bula de {dict_row['nome']} "

                for idx, pergunta in enumerate(list_perguntas):

                    if (idx == 8):
                        prompt = clean_paciente_last_question_prompt.format(text=text, pergunta=pergunta)
                    else:
                        prompt = clean_paciente_prompt.format(text=text, pergunta=pergunta, pergunta_anterior=list_perguntas[idx+1])               
                    answer = model.inference(prompt)
                    dict_row[pergunta] = answer.replace('\n', ' ')
                    dict_row['full_topic'] = dict_row['full_topic'] + f" {pergunta} {dict_row[pergunta]}"          
                
            df = df._append(dict_row, ignore_index=True)

        except Exception as error:
            atualizar_log_erro(error)
            continue          
           
        # Saving in csv file
        df.to_csv(config['result_file'], index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("The configuration file is missing.")
    else:
        config_file = sys.argv[1]
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            main(config)