import os
import yaml
import PyPDF2

import pandas as pd
from tqdm import tqdm
from models.llama3 import Llama3
from legacy.gemini_model import GeminiModel

from topics import list_perguntas

# list_perguntas = [
#     "1. Para que este medicamento é indicado?",
#     "2. COMO ESTE MEDICAMENTO FUNCIONA? ",
#     "3. Quando não devo usar este medicamento?",
#     "4. O QUE DEVO SABER ANTES DE USAR ESTE MEDICAMENTO?",
#     "5. Onde, como e por quanto tempo posso guardar esse medicamento?",
#     "6. Como devo usar esse medicamento?",
#     "7. O que devo fazer quando eu me esquecer de usar esse medicamento?",
#     "8. Quais os males que este medicamento pode me causar?",
#     "9. O que fazer se alguém usar uma quantidade maior do que a indicada deste medicamento?",
# ]

# list_topics = [
#     "COMPOSIÇÃO",
#     "1. INDICAÇÕES",
#     "2. RESULTADOS DE EFICÁCIA",
#     "3. CARACTERÍSTICAS FARMACOLÓGICAS",
#     "4. CONTRAINDICAÇÕES",
#     "5. ADVERTÊNCIAS E PRECAUÇÕES",
#     "6. INTERAÇÕES MEDICAMENTOSAS",
#     "7. CUIDADOS DE ARMAZENAMENTO DO MEDICAMENTO",
#     "8. POSOLOGIA E MODO DE USAR",
#     "9. REAÇÕES ADVERSAS",
#     "10. SUPERDOSE"
# ]

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

    df = pd.DataFrame(columns=['nome', 'id', 'pergunta 1', 'pergunta 2', 'pergunta 3', 'pergunta 4', 
                           'pergunta 5', 'pergunta 6', 'pergunta 7', 'pergunta 8', 
                           'pergunta 9', ])
    
    #model = GeminiModel()
    model = Llama3()
    paciente_path = '/home/navarro/bulas/bulas_unicas/paciente/'
    lst_max_tokens = []

    for filename in tqdm(os.listdir(paciente_path)):

        dict_row = {}

        # print(filename)
        text = extract_text_from_pdf(paciente_path + filename)
        tokens = model.count_tokens(text)
        # print(f"Tokens: {tokens}")

        if tokens > 7500:
            lst_max_tokens.append(filename)
            continue

        nome_medicamento = model.inference(f"Given this leaflet: {text}, what is the name of the medicine? Answer only the name.")
        dict_row['nome'] = nome_medicamento
        dict_row['id'] = filename.replace('.pdf','')

        for pergunta in list_perguntas:

            if (list_perguntas.index(pergunta) == 8):
                prompt = f""" Given this leaflet:
                {text}
                Return the topic with the title: {pergunta}
                A topic only ends when another begins, so return all text until next topic coming. 
                Do not use line breaks.
                Don't summarize.
                Fix spelling errors if there are any.
                Do not return the topic title, just the content.
                If the topic is not in context, reply 'NaN'
                """
            else:
                prompt = f""" Given this leaflet:
                {text}
                Return the topic with the title: {pergunta}
                A topic only ends when another begins, so return all text until next topic ({list_perguntas[list_perguntas.index(pergunta) + 1]}) coming. 
                Do not use line breaks.
                Don't summarize.
                Fix spelling errors if there are any.
                Do not return the topic title, just the content.
                If the topic is not in context, reply 'NaN'
                """                
            answer = model.inference(prompt)
            dict_row[f'pergunta {list_perguntas.index(pergunta) + 1}'] = answer.replace('\n', ' ')


        df = df._append(dict_row, ignore_index=True)

    # Saving in csv file
    df.to_csv('paciente.csv')

    # Saving filenames that have exceeded the maximum number of llama tokens 3
    with open('lst_max_tokens_paciente.txt', 'w') as file:
        for item in lst_max_tokens:
            file.write(item + "\n")


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
