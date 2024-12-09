from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
   
    def count_tokens(self, prompt):
        tokens = self.tokenizer(prompt)
        return len(tokens['input_ids'])



    def inference(self, prompt, system_prompt = None):

        messages = [            
            {"role": "user", "content": prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 300,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        return (output[0]['generated_text'])

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