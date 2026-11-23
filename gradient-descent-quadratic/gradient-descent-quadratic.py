def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    y = 0
    while steps:
        x0 = x0 - lr*(2*a*x0 + b)
        y = a*x0*x0+b*x0+c
        steps = steps -1
    return x0
        
        