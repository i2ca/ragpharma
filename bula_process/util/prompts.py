clean_profissional_prompt = """ Given this leaflet:
{parte}
Return the topic with the title: {topic_name} or, return 'Nan'
A topic only ends when another begins.
Do not use line breaks
Don't summarize.
Please respond only in Brazilian Portuguese.
Fix spelling errors if there are any.
Do not return the topic title, just the content.               
"""

clean_paciente_last_question_prompt = """ Given this leaflet:
{text}
Return the topic with the title: {pergunta}
A topic only ends when another begins, so return all text until next topic coming. 
Do not use line breaks.
Don't summarize.
Fix spelling errors if there are any.
Do not return the topic title, just the content.
If the topic is not in context, reply 'NaN'
"""

clean_paciente_prompt = """ Given this leaflet:
{text}
Return the topic with the title: {pergunta}
A topic only ends when another begins, so return all text until next topic ({pergunta_anterior}) coming. 
Do not use line breaks.
Don't summarize.
Fix spelling errors if there are any.
Do not return the topic title, just the content.
If the topic is not in context, reply 'NaN'
"""  

prompt_check_question = """Given this medicine leaflet:
{context}
And this question: {json_question}
Check if this alternative answers this question:
{wrong_answer}
Return ONLY "YES" if it is the right answer and "NO" if it is not the answer.
"""

prompt_check_context = """Given this medicine leaflet:
{context}
and this question:
{question}
Check if this answer is in the context:
{answer}
Return ONLY "YES" if it is the right answer and "NO" if it is not the answer.
"""

prompt_points = """Given this medicine leaflet:
{context}
Return in brazilian portuguese the principal points.
"""

system_prompt_question = "You are a strict but fair teacher who raises questions about medicine leaflets."

prompt_make_question = """Given this medicine leaflet:
{context}
Create a multiple-choice question about {assunto} from this section:
{section}
The question must have one right answer and three wrong answers
Include the name of the medicine {nome_remedio} in the question.
Do not generalize. Make question specific from the medicine. Don't make obvious questions.
The right answer must be in the context, but wrong answers do not necessarily need to be.
The right answer cannot be bigger than the wrong options! All the choices must have similar size.
Ask difficult questions.
{another_question}
Return in json format with:
{{"query": question, "gold_choice": correct answer, "wrong_choices": list of wrong answers}}.
Examples:
{{"query": "Qual das seguintes indicações é tratada por Secni?", "gold_choice": "Amebíase intestinal", "wrong_choices": ["Artrite reumatóide", "Doença cardiovascular", "Hipertensão"] }}
{example}
Reply without markdown and without line breaks.""" 

prompt_create_new_wrong_answers = """
Given this question in json format,
{question_json}
rewrite the wrong choices to make them similar to the gold, but still wrong. Return in the same format.
return ONLY the changed wrong choices in json format. without breaklines.
Format:
{"wrong_choices":[list of wrong choices]}
"""
# {{"query": "Qual é a dose inicial recomendada para o tratamento da depressão com o Cloridrato de fluoxetina?", "gold_choice": "20 mg/dia (20 gotas)", "wrong_choices": ["40 mg/dia (40 gotas)", "10 mg/dia (10 gotas)", "80 mg/dia (80 gotas)"] }}
