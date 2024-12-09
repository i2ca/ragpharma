import os
import sys
import yaml
from tqdm import tqdm
from models import Phi
from util.error_log import atualizar_log_erro
from clean_process import PDFSectionProcessor, PDFTextExtractor

def main(config):
    # Limpando o arquivo de log
    if os.path.isfile('log.txt'):
        os.remove('log.txt')

    pdfs_path = config['pdfs_path']
    model = Phi()
    processor = PDFSectionProcessor(model, config)

    for filename in tqdm(os.listdir(pdfs_path)[:500]):
        try:
            pdf_extractor = PDFTextExtractor(pdfs_path + filename)
            text = pdf_extractor.extract_text()
            dict_row = processor.process_pdf(text, filename)
            processor.save_to_csv(dict_row, config['result_file'])  # Save to CSV after processing each PDF
        except Exception as error:
            atualizar_log_erro(error)
            continue

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("The configuration file is missing.")
    else:
        config_file = sys.argv[1]
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            main(config)
