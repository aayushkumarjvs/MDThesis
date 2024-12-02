import torch

def memorization_loss(model_output, target_output):
    """
    Loss function to penalize memorization. 
    Compares the similarity between the generated output and the target to ensure novelty.
    """
    similarity = torch.cosine_similarity(model_output, target_output, dim=1)
    return torch.mean(similarity)

def custom_loss_fn(model_output, target_output, memorization_weight=0.1):
    """
    Combines standard loss (e.g., CrossEntropy) with memorization loss.
    """
    ce_loss = torch.nn.CrossEntropyLoss()(model_output, target_output)
    mem_loss = memorization_loss(model_output, target_output)
    total_loss = ce_loss + memorization_weight * mem_loss
    return total_loss
