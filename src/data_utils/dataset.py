from src.data_utils.document import Document
from torch.utils.data import Dataset
import random

class DocumentDataset:
    def __init__(self, documents: list[dict]) -> None:
        self.documents: list[Document] = self.__BuildDocuments(documents)
        
    def __BuildDocuments(self, documents):
        
        return [Document(text=text, _id=_id) for entries in documents 
                    for (text, _id) in zip(entries['text'], entries['document_id'])
                ]
        
    def GetDocuments(self) -> list[Document]:
        return self.documents

class DPRDataset(Dataset):
    def __init__(self,split):
        self.questions = []
        self.texts = []

        for question, texts in zip(split["question"], split["text"]):

            if isinstance(texts, str):
                texts = [texts]

            texts = [text for text in texts if text.strip()]

            if texts:
                self.questions.append(question)
                self.texts.append(texts)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        texts = self.texts[idx]

        # Randomly select a text from the list of texts for this question
        text = random.choice(texts)

        return question, text