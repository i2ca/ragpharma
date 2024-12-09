import os
import pandas as pd
from util.prompts import clean_profissional_prompt, clean_paciente_last_question_prompt, clean_paciente_prompt
from util.topics import list_perguntas, list_topics

class PDFSectionProcessor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def process_profissional(self, text, filename):
        dict_row = {}
        full_size = len(text)
        half = full_size // 2
        halfhalf = half // 2
        p1 = text[:half]
        p2 = text[halfhalf:full_size - halfhalf]
        p3 = text[half:]

        tokens_p1 = self.model.count_tokens(p1)

        if tokens_p1 > 7200:
            raise Exception(f"{filename} has more than 7500 tokens in a half file.")

        list_ptr = [p1, p2, p3]
    
        nome_medicamento = self.model.inference(f"Given this leaflet: {p1}, what is the name of the medicine? Answer only the name.")
        dict_row['nome'] = nome_medicamento.replace('.', '')
        dict_row['id'] = filename.replace('.pdf','')
        dict_row['full_topic'] = f"Bula de {dict_row['nome']} "

        for idx, parte in enumerate(list_ptr):         
            for topic_name in list_topics[idx]:                           
                loaded_prompt = clean_profissional_prompt.format(parte = parte, topic_name = topic_name)
                topic = self.model.inference(loaded_prompt)
                dict_row[topic_name] = topic.replace('\n', ' ')
                dict_row['full_topic'] = dict_row['full_topic'] + f" {topic_name} {dict_row[topic_name]}"

        return dict_row

    def process_paciente(self, text, filename):
        dict_row = {}
        tokens = self.model.count_tokens(text)
        if tokens > 7500:
            raise Exception(f"{filename} has more than 7500 tokens.")

        nome_medicamento = self.model.inference(f"Given this leaflet: {text}, what is the name of the medicine? Answer only the name.")
        dict_row['nome'] = nome_medicamento
        dict_row['id'] = filename.replace('.pdf','')
        dict_row['full_topic'] = f"Bula de {dict_row['nome']} "

        for idx, pergunta in enumerate(list_perguntas):
            if (idx == 8):
                prompt = clean_paciente_last_question_prompt.format(text=text, pergunta=pergunta)
            else:
                prompt = clean_paciente_prompt.format(text=text, pergunta=pergunta, pergunta_anterior=list_perguntas[idx+1])               
            answer = self.model.inference(prompt)
            dict_row[pergunta] = answer.replace('\n', ' ')
            dict_row['full_topic'] = dict_row['full_topic'] + f" {pergunta} {dict_row[pergunta]}" 

        return dict_row
    
    def process_pdf(self, text, filename):
        if self.config['type'] == 'profissional':
            return self.process_profissional(text, filename)
        elif self.config['type'] == 'paciente':
            return self.process_paciente(text, filename)
        else:
            raise Exception("Config type not found. Choose between 'profissional' and 'paciente'.")

    def save_to_csv(self, dict_row, result_file):
        # Converte o dicionário em um DataFrame de uma linha
        df = pd.DataFrame([dict_row])
        
        # Salva a linha no arquivo CSV, sem sobrescrever o arquivo inteiro
        if not os.path.isfile(result_file):
            df.to_csv(result_file, index=False)  # Salva com cabeçalho se o arquivo não existir
        else:
            df.to_csv(result_file, mode='a', header=False, index=False)  # Acrescenta a linha sem cabeçalho
