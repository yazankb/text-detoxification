
import torch
from sklearn import metrics
from tqdm import tqdm
import transformers
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import numpy as np
from functools import partial

checkpoint_dir = "../models"

def TrainOneEpochToxicTextClassifier(model, dataloader, device, loss_fn, optimizer, epoch):
    """
    Train the toxic text classifier for one epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for the training data.
        device: The device (CPU or GPU) on which to perform training.
        loss_fn: The loss function used for training.
        optimizer: The optimizer used for updating model parameters.
        epoch: The current epoch number.

    Returns:
        float: The total loss for the epoch.
    """

    model.train()
    epoch_loss = 0 
    for _,data in tqdm(enumerate(dataloader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        epoch_loss+= loss.item()
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return epoch_loss


def ValidateToxicTextClassifier(model, dataloader, device, best_f1, save_name = '', training = False):
    """
    Validate the toxic text classifier on a validation dataset.

    Args:
        model: The PyTorch model to validate.
        dataloader: DataLoader for the validation data.
        device: The device (CPU or GPU) on which to perform validation.
        best_f1: The best f1 validation score until this epoch
        save_name(str): The name of the saved checkpoint after finishing training
        training(bool): training mode (to know if the model should be saved after validation is done)
    Returns:
        dict: A dictionary containing validation metrics (accuracy, macro F1).
    """

    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            
    fin_outputs = (np.array(fin_outputs) >= 0.5).tolist()
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')
    if f1_score_macro > best_f1 and training:
        model.save_pretrained(checkpoint_dir + checkpoint_name)

    return {'accuracy' : accuracy, 'f1_score_macro' : f1_score_macro}

def TrainToxicTextClassifier(model, train_dataloader, validate_dataloader, device, loss_fn, optimizer, epochs, save_name):
    """
    Train and validate the toxic text classifier for multiple epochs.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        validate_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        device (str): The device ('cpu' or 'cuda') on which to perform training and validation.
        loss_fn: The loss function used for training.
        optimizer: The optimizer used for updating model parameters.
        epochs (int): The number of training epochs.
        save_name(str): The name of the saved checkpoint after finishing training
    Returns:
        dict: A dictionary containing the training loss and validation metrics.
            {
                'loss_list': List of training loss for each epoch,
                'metrics_list': List of validation metrics for each epoch,
            }
    """

    loss_list = []
    metrics_list = []
    best_f1 = 0
    for epoch in tqdm(range(epochs)):
        epoch_loss = TrainOneEpochToxicTextClassifier(model, train_dataloader, device, loss_fn, optimizer, epoch)
        loss_list.append(epoch_loss)
        ret = ValidateToxicTextClassifier(model, validate_dataloader, device, best_f1, save_name, True)
        metrics_list.append(ret)
        best_f1 = ret['f1_score_micro']
    return {'loss_list' : loss_list, 'metrics_list' : metrics_list}



class BERTClass(torch.nn.Module):
    """
    A PyTorch module for fine-tuning a pre-trained BERT model for binary classification tasks.

    Args:
        checkpoint (str): The name of the pre-trained BERT model checkpoint to load.
        
    Attributes:
        l1 (transformers.BertModel): The pre-trained BERT model.
        l2 (torch.nn.Linear): A linear layer for binary classification (output dimension: 1).

    Methods:
        forward(ids, mask, token_type_ids):
            Forward pass of the BERT model for binary classification.

    Example:
        # Create an instance of BERTClass
        model = BERTClass("bert-base-uncased")

        # Forward pass
        outputs = model(ids, mask, token_type_ids)
    """

    def __init__(self, checkpoint):
        """
        Initializes a BERTClass instance.

        Args:
            checkpoint (str): The name of the pre-trained BERT model checkpoint to load.
        """
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(checkpoint)
        self.l2 = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask, token_type_ids):
        """
        Forward pass of the BERT model for binary classification.

        Args:
            ids (torch.Tensor): Input token IDs.
            mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type IDs.

        Returns:
            torch.Tensor: Model output for binary classification.
        """
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.l2(output_1)
        return output



def compute_metrics(eval_pred , tokenizer):
    """
    Compute evaluation metrics for a Seq2Seq model.

    Args:
        eval_pred (tuple): A tuple containing predictions and labels.
        tokenizer:   The tokenizer used to process the text before.

    Returns:
        dict: A dictionary containing computed evaluation metrics.
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def TrainTextDetoxificationModel(model, datasets, tokenizer, epochs, lr, weight_decay, batch_size=16, checkpoint_dir=checkpoint_dir):
    """
    Train a text detoxification model using the provided configurations and datasets.

    Args:
        model (transformers.PreTrainedModel): The pre-trained Seq2Seq model to fine-tune.
        datasets (dict): A dictionary containing train and validation datasets.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for processing text.
        epochs (int): The number of training epochs.
        lr (float): The learning rate for training.
        weight_decay (float): The weight decay for training.
        batch_size (int, optional): The batch size for training and evaluation (default: 16).
        checkpoint_dir (str, optional): The directory to save model checkpoints (default: checkpoint_dir).

    Returns:
        dict: A dictionary containing training logs and metrics.

    Example:
        # Define the training and validation datasets
        datasets = {
            "train": train_dataset,
            "validation": validation_dataset
        }

        # Define the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

        # Train the model
        logs = TrainTextDetoxificationModel(model, datasets, tokenizer, epochs=3, lr=2e-5, weight_decay=0.01)
    """
    args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy = "steps",
        eval_steps=2000,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16 = True,
        push_to_hub=False,
        overwrite_output_dir = True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics_lambda = partial(compute_metrics, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_lambda,
    )

    trainer.train()
    trainer.save_model(checkpoint_dir)
    return trainer.state.log_history
