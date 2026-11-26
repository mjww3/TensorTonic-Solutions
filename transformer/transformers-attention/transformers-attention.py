import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_model = Q.shape[-1]
    scaled_scores = (( Q@K.transpose(-2,-1))/math.sqrt(d_model))
    attention_weights = torch.softmax(scaled_scores,dim=-1)
    attention_output = attention_weights@V
    return attention_output
    
    
    