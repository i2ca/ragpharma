from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

import torch

class Phi():
    def __init__(self) -> None:
        
        model_id = "microsoft/Phi-3-small-8k-instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True, 
        )
   
    def count_tokens(self, prompt):
        tokens = self.tokenizer(prompt)
        return len(tokens['input_ids'])



    def inference(self, prompt, system_prompt = None):

        if not system_prompt:
            system_prompt = "You are a helpful assistant who only answers the questions asked."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to('cuda')
        self.model.to('cuda')

        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        
        torch.cuda.empty_cache()
        return decoded[0]
        
    def perplexity(self, prompt, answer):
        # self.model.to('cuda')
        inputs_length = len(self.tokenizer(prompt, return_tensors="pt").to("cuda")["input_ids"][0])
        complete_phrase = prompt + answer       
        input = self.tokenizer(complete_phrase, return_tensors="pt").input_ids.to("cuda")
        target_id = input.clone()
        target_id[:,:inputs_length] = -100

        with torch.no_grad():
            outputs = self.model(input, labels=target_id)
            neg_log_likelihood = outputs.loss
        
        ppl = torch.exp(neg_log_likelihood)
        torch.cuda.empty_cache()
        return ppl.item()   