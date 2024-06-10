def atualizar_log_erro(erro_msg, arquivo_log='log.txt'):
    try:
        with open(arquivo_log, 'a') as arquivo:
            arquivo.write(f'{erro_msg}\n')
    except Exception as e:
        print(f"Erro ao atualizar o arquivo de log: {e}")
