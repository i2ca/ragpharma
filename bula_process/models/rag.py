import pandas as pd
import torch
import numpy as np
import sentence_transformers 

from sentence_transformers import SentenceTransformer

class Rag():
    def __init__(self) -> None:
        

        df = pd.read_csv('datasets/mini_paciente_embbeds.csv')
        df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

        self.embbeds = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to('cpu')
        self.data = df.to_dict(orient="records")

        # Obviously cuda could be a faster and better option, but we prioritize the GPU for LLM inference.
        self.embedding_model = SentenceTransformer(model_name_or_path="intfloat/multilingual-e5-small", device="cpu")

    def retrieve(self,
                 query: str):
        query_embedding = self.embedding_model.encode(query, 
                                    convert_to_tensor=True)
        dot_scores = sentence_transformers.util.dot_score(query_embedding, self.embbeds)[0]
        scores, indices = torch.topk(input=dot_scores,k=1)
        context = self.data[indices]["full_topic"]
        name_medication_rag = self.data[indices]["id"]
        # print(self.data[indices]["nome"])
        # print(self.data[indices]["full_topic"])
        return context, name_medication_rag


