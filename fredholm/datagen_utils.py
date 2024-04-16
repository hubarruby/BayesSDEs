import numpy as np

def create_data(diffusion, b_func, t_delta, t_end, start_val, verbose=False, **kwargs):
    """
    Simulates and returns a sequence of data points representing the evolution of a system over time, based on a specified deterministic function and a stochastic diffusion component.
    The simulation omits the initial 5% of data points to focus on the stabilized behavior of the system.

    Parameters:
    - diffusion (float): Intensity of the stochastic fluctuation component.
    - b_func (callable): Deterministic function that defines the system's evolution, accepting the current system value and optional keyword arguments.
    - t_delta (float): Time increment for each simulation step.
    - t_end (float): Total duration of the simulation.
    - start_val (float): Initial value of the system.
    - **kwargs: Additional keyword arguments passed to the deterministic function.

    Returns:
    - list of float: A list of simulated system values after the initial transient phase.
    """
    t = 0
    val = start_val
    data = []

    def create_next(val):
        return val + b_func(val, **kwargs) * t_delta + diffusion * np.random.normal(0, np.sqrt(t_delta))

    while t < t_end - t_delta:
        data.append(val)
        val = create_next(val)
        t += t_delta
        if verbose: # and (t-int(t) == 0 or t-int(t)==0.5):
            print(f'Generated data up to t = {t}/{t_end}')

    return data[len(data) // 20:]


# b for our first example, 3.1
def b1(x, y):
    return (x - x ** 3) * y


# defining b for the second example, 3.2
def b2(x, y):
    return (x + y) - (x + y) ** 3


# calculation of the integral
def b_bar(x, b, pi, integral_N=1000):
    samples = pi.rvs(size=integral_N)
    # print(samples.shape)
    # print(b(x, samples).shape)
    # print(np.sum(b(x, samples)))
    return np.sum(b(x, samples)) / integral_N


# b for our first example, 3.1
def b1_v(x, y):
    """
    Computes the value of b for each combination of elements in vectors (or numpy arrays) of x and y values,
    or a single value of x and y, according to the formula (x - x**3) * y. The result is a matrix where each
    row corresponds to an x and each column to a y, or a single value if x and y are both scalars.

    Parameters:
    - x (numpy.array or float): A vector (numpy array) of x values or a single float value.
    - y (numpy.array or float): A vector (numpy array) of y values or a single float value.

    Returns:
    - numpy.array or float: A 2D array where each element [i, j] is the result of (x[i] - x[i]**3) * y[j],
      or a single value if x and y are both scalars.
    """
    # Check if x and y are scalars
    if np.isscalar(x) and np.isscalar(y):
        # Direct computation for scalars
        result = (x - x**3) * y
    else:
        # Ensure x and y are numpy arrays for broadcasting
        x = np.asarray(x)
        y = np.asarray(y)

        # Check dimensions and reshape x to a column vector for broadcasting if necessary
        if np.ndim(x) == 1:
            x = x[:, np.newaxis]

        # Apply the formula
        result = (x - x**3) * y

    return result

# defining b2 from the second exmaple, 3.2
def b2_v(x, y):
    """
    Computes the value of the expression (x + y) - (x + y)**3 for each combination of elements
    in vectors (or numpy arrays) of x and y values, or a single value of x and y. The result is
    a matrix where each row corresponds to an x and each column to a y, or a single value if x
    and y are both scalars.

    Parameters:
    - x (numpy.array or float): A vector (numpy array) of x values or a single float value.
    - y (numpy.array or float): A vector (numpy array) of y values or a single float value.

    Returns:
    - numpy.array or float: A 2D array where each element [i, j] is the result of the expression
      (x[i] + y[j]) - (x[i] + y[j])**3 for each pair of x[i] and y[j], or a single value if x and
      y are both scalars.
    """
    # Check if x and y are scalars
    if np.isscalar(x) and np.isscalar(y):
        # Direct computation for scalars
        result = (x + y) - (x + y)**3
    else:
        # Ensure x and y are numpy arrays for broadcasting
        x = np.asarray(x)
        y = np.asarray(y)

        # Broadcasting x and y to form all pairs (x[i] + y[j])
        x_expanded = x[:, np.newaxis] if x.ndim == 1 else x
        y_expanded = y[np.newaxis, :] if y.ndim == 1 else y

        # Applying the expression to each pair
        result = (x_expanded + y_expanded) - (x_expanded + y_expanded)**3

    return result


# calculation of the integral
def b_bar_v(x, b, pi, integral_N=1000):
    samples = pi.rvs(size=integral_N)
    # print(samples.shape)
    # print(b(x, samples).shape)
    # print(np.sum(b(x, samples)))
    return np.sum(b(x, samples), axis=1) / integral_N
