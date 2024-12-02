import torch
import nltk

def compute_coherence(pred, target):
    """Measures text fluency and logical consistency"""
    # Using NLTK for sentence coherence
    sentences = nltk.sent_tokenize(pred)
    coherence_score = 0
    for i in range(len(sentences)-1):
        coherence_score += measure_sentence_similarity(sentences[i], sentences[i+1])
    return coherence_score / max(len(sentences)-1, 1)

def compute_relevance(pred, target):
    """Measures semantic similarity between prediction and target"""
    pred_embedding = get_embedding(pred)
    target_embedding = get_embedding(target)
    return cosine_similarity(pred_embedding, target_embedding)

def compute_accuracy(pred, target):
    """Measures exact match between prediction and target"""
    return 1.0 if pred.strip() == target.strip() else 0.0

def compute_memory_savings(original_size, quantized_size):
    """Calculates memory reduction percentage"""
    return (original_size - quantized_size) / original_size * 100

def evaluate_model(model, tokenizer, dataset):
    """
    Evaluate model performance on coherence, relevance, and accuracy.
    """
    scores = {"coherence": [], "relevance": [], "accuracy": []}
    
    for sample in dataset:
        inputs = tokenizer(sample['input'], return_tensors="pt").input_ids
        outputs = model.generate(inputs)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Coherence Score
        scores["coherence"].append(compute_coherence(decoded_output, sample["target"]))
        # Relevance Score
        scores["relevance"].append(compute_relevance(decoded_output, sample["target"]))
        # Accuracy
        scores["accuracy"].append(compute_accuracy(decoded_output, sample["target"]))

    return {metric: sum(scores[metric])/len(scores[metric]) for metric in scores}
