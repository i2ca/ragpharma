from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

import torch

class Llama3():
    def __init__(self) -> None:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        access_token = "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"
        logging.set_verbosity_error() 
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=access_token
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

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        with torch.inference_mode() and torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        
            response = outputs[0][input_ids.shape[-1]:]
            torch.cuda.empty_cache()
            return self.tokenizer.decode(response, skip_special_tokens=True)
        
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