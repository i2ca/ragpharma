import os
import yaml
import PyPDF2

import pandas as pd
from models.llama3 import Llama3
from legacy.gemini_model import GeminiModel
from tqdm import tqdm

from topics import list_topics
from util.prompts import clean_profissional_prompt


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
    # Config the api key and select the model

    df = pd.DataFrame()
    
    #model = GeminiModel()
    model = Llama3()
    paciente_path = '/home/navarro/bulas/bulas_unicas/profissional/'
#    paciente_path = 'bulas/'
    lst_max_tokens = []

    # Carregando os IDs do arquivo mini_bulario_ids.txt
    with open('datasets/mini_bulario_ids.txt', 'r') as file:
        ids = file.read().splitlines()

    for id in tqdm(ids[:2]):

        numero_formatado = "{:0>{}}".format(id, 10)
        filename = f"/home/navarro/bulas/bulas_unicas/profissional/{numero_formatado}.pdf"
        dict_row = {}

        # print(filename)
        text = extract_text_from_pdf(filename)

        full_size = len(text)
        half = full_size // 2
        halfhalf = half // 2
        p1 = text[:half]
        p2 = text[halfhalf:full_size - halfhalf]
        p3 = text[half:]

        tokens_p1 = model.count_tokens(p1)

        if tokens_p1 > 7200:
            continue
        list_ptr = [p1, p2, p3]

        nome_medicamento = model.inference(f"Given this leaflet: {p1}, what is the name of the medicine? Answer only the name.")
        dict_row['nome'] = nome_medicamento.replace('.', '')
        dict_row['id'] = id
        dict_row['full_topic'] = f"Bula de {dict_row['nome']} "
        # index_pergunta = 1
        for idx, parte in enumerate(list_ptr):          

            print(list_topics[idx])
            for topic_name in list_topics[idx]:
                
                
                loaded_prompt = clean_profissional_prompt.format(parte = parte, topic_name = topic_name)
                topic = model.inference(loaded_prompt)

                # print(pergunta)
                # print(answer)
                # print('\n')
                # dict_row[f'topico {index_pergunta}'] = answer.replace('\n', ' ')
                dict_row[topic_name] = topic.replace('\n', ' ')
                dict_row['full_topic'] = dict_row['full_topic'] + f" {topic_name} {dict_row[topic_name]}"
                # index_pergunta = index_pergunta + 1


        df = df._append(dict_row, ignore_index=True)

    # Saving in csv file
    df.to_csv('profissional.csv', index=False)

    # # Saving filenames that have exceeded the maximum number of llama tokens 3
    # with open('lst_max_tokens.txt', 'w') as file:
    #     for item in lst_max_tokens:
    #         file.write(item + "\n")


if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
