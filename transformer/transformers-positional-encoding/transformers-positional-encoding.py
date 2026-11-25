import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pe = np.zeros((seq_length,d_model))
    for pos in range(seq_length):
        for index in range(d_model):
            i = index//2
            angle = pos/(10000**((2*i)/d_model))
            if index%2 == 0:
                pe[pos,index] = np.sin(angle)
            else:
                pe[pos,index] = np.cos(angle)
    return pe
        