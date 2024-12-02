from datasets import load_dataset

def load_imhi_dataset():
    """
    IMHI dataset from MentalLLaMA's repository.
    """
    dataset = load_dataset("MentalLLaMA/IMHI")
    return dataset
