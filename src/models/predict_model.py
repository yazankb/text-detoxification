import torch
from transformers import BertForSequenceClassification, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer


def SampleSelectBest(model_classifier, tokenizer_classifier, model_generator, tokenizer_generator, input_text, num_samples=5):
    """
    Generate multiple samples using the T5 model and select the sample with the highest classification score using the BERT model.

    Args:
        model_classifier (str): Pre-trained BERT model for sequence classification.
        tokenizer_classifier (str): Tokenizer corresponding to the BERT model.
        model_generator (str): Pre-trained T5 model for conditional text generation.
        tokenizer_generator (str): Tokenizer corresponding to the T5 model.
        input_text (str): The input text for which samples are generated.
        num_samples (int): Number of samples to generate from the T5 model.

    Returns:
        best_sample (str): The sample text with the highest classification score.
    """
    inputs = tokenizer_classifier(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    samples_and_scores = []

    for _ in range(num_samples):
        # Generate a sample using the T5 model
        input_ids = tokenizer_generator.encode(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        output = model_generator.generate(input_ids, max_length=128, num_return_sequences=1, do_sample=True)

        generated_text = tokenizer_generator.decode(output[0], skip_special_tokens=True)

        generated_inputs = tokenizer_classifier(generated_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            logits = model_classifier(**generated_inputs).logits
            scores = torch.softmax(logits, dim=1)[0]

        classification_score = scores[0].item()

        samples_and_scores.append((generated_text, classification_score))

    # Sort the generated samples based on their classification scores in descending order
    samples_and_scores.sort(key=lambda x: x[1], reverse=True)

    # Select the sample with the highest classification score
    best_sample = samples_and_scores[0][0]

    return best_sample