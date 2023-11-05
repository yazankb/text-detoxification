import pandas as pd
import torch
from torch.utils.data import Dataset

# Define labels
labels = ["nontoxic", "toxic"]

# Create index-to-label dictionary
index_to_label = {i: label for i, label in enumerate(labels)}

# Create label-to-index dictionary
label_to_index = {label: i for i, label in enumerate(labels)}

class ToxicTextClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for toxic text classification.
    
    """
    
    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initializes the dataset.

        Args:
            dataframe (DataFrame): A Pandas DataFrame containing text data and labels.
            tokenizer: The tokenizer for encoding text.
            max_len (int): Maximum sequence length for padding/truncating text.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Creating a sentence-label pair to be used for training 
        self.sentences = pd.concat([dataframe['translation'],
                                    dataframe['reference']]).reset_index(drop=True)
        self.labels = pd.concat([dataframe['trn_tox'].round(),
                                 dataframe['ref_tox'].round()]).reset_index(drop=True)
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.sentences)
    
    def __getitem__(self, index):
        """
        Returns a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input IDs, attention mask, token type IDs, and labels.
        """
        sentence = str(self.sentences[index])
        sentence = " ".join(sentence.split())

        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.labels[index], dtype=torch.float)
        }
    


class TextDetoxificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initialize the TextDetoxificationDataset.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the dataset.
            tokenizer: The tokenizer used to preprocess the text.
            max_len (int): The maximum length for tokenized sequences.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Separate toxic and non-toxic samples from the DataFrame
        self.toxic = pd.concat([dataframe['translation'][dataframe['trn_tox'] > dataframe['ref_tox']], dataframe['reference'][dataframe['ref_tox'] > dataframe['trn_tox']]])#.reset_index(drop=True)
        self.nontoxic = pd.concat([dataframe['translation'][dataframe['trn_tox'] < dataframe['ref_tox']], dataframe['reference'][dataframe['ref_tox'] < dataframe['trn_tox']]])#.reset_index(drop=True)
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.toxic)
    
    def __getitem__(self, index):
        """
        Returns a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input IDs, attention_mask, and labels.
        """
        input = str(self.toxic[index])

        # Tokenize the input text
        model_input = self.tokenizer(input, max_length=self.max_len, truncation=True)

        # Tokenize the labels (non-toxic text)
        labels = self.tokenizer(text_target=str(self.nontoxic[index]), max_length=self.max_len, truncation=True)

        # Add labels to the model_input dictionary
        model_input['labels'] = labels['input_ids']

        return model_input
