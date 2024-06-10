import google.generativeai as genai
import google.ai.generativelanguage as genl
from google.generativeai import GenerationConfig
# from vertexai import generative_models
from time import sleep

class GeminiModel():
    def __init__(self) -> None:
        id = 'AIzaSyCdQlVl_eGEMg0yDwpxOGsG_wBiqBotQwA'
        type_model = 'models/gemini-pro'

        genai.configure(api_key=id)
        self.gemini = genai.GenerativeModel(type_model)
        # Generation Config
        self.config = GenerationConfig(
            max_output_tokens=1024, 
        )
        # Safety config
        self.safety_config = {
            genl.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    def count_tokens(self, prompt):
        return self.gemini.count_tokens(prompt)

    def inference(self, prompt):
        answer = ''
        for x in range(0, 4):  # try 4 times
            
            try:
                # answer = self.gemini.generate_content(prompt).text
                response = self.gemini.generate_content(prompt, generation_config=self.config, safety_settings=self.safety_config)
                if response.candidates:
                    if response.candidates[0].content.parts:
                        # print(response.candidates[0].content.parts[0].text)
                        # print(response.text)
                        answer = response.candidates[0].content.parts[0].text
                    else:
                        raise Exception('Erro na geração.' + prompt)
                else:
                    raise Exception('Erro na geração.' + prompt)

                # # print(response.prompt_feedback)
                # if len(response.candidates[0].safety_ratings) == 4:
                #     if response.candidates[0].safety_ratings[3] != 'category: HARM_CATEGORY_DANGEROUS_CONTENT \nprobability: NEGLIGIBLE':
                #         print(response.candidates[0].safety_ratings[3])
                #         print(response.candidates[0].content.parts[0].text)
                #     if response.text:
                #         answer = response.text
                #     else:                       
                #         raise Exception(response.prompt_feedback)
                # else:
                #     raise Exception('Erro na geração.')

                str_error = None
            except Exception as e:
                print(str(e))
                if x == 1:
                    prompt = prompt.replace('este','esse')
                    prompt = prompt.replace('deste','desse')

                elif x == 2:
                    prompt = prompt.replace('esse','este')       
                    prompt = prompt.replace('desse','deste')                    
             
                str_error = str(e)
                # if response.prompt_feedback:
                #     print("Feedback:")
                #     print(response.prompt_feedback)
                # print(str_error)
                # print(str_error)
                # candidates = response.candidates
                # if len(candidates) > 0:
                #     if len(candidates[0].content.parts) > 0:
                #         print("TEXTO GERADO: " + candidates[0].content.parts[0].text)
                #         print(candidates[0].safety_ratings)
                
                pass

            if str_error:
                sleep(5)  # wait for 5 seconds before trying to fetch the data again
            else:
                break
        if answer == '':
            print("DEU RUIM POR AQUI")

        return answer
        
