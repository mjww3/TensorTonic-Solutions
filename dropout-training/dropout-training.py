import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    if not rng:
        rng = np.random.default_rng()
    x = np.asarray(x)
    random = rng.random(x.shape)
    mask = (random <= (1-p)).astype(float)
    mask = mask * (1/(1-p))
    x = x * mask
    return (x,mask)