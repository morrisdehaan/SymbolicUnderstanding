import torch
import numpy as np
import sympy
import omegaconf
from typing import Literal, Tuple, Union, Dict

def sample_equation(
    eq: str, vars_used: list[str], vars_model: list[str],
    point_count: int,
    min_support: float, max_support: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly samples an equation using `sympy.lambdify`.

    # Args
    * `vars_used`: The names of the variables in the equation.
    * `vars_model`: The total list of names of the variables the model expects. If the equation has less variables,
        the input will be padded with zeros.
    * `min_support` & `max_support`: The range of the domain to sample the equation from.

    # Returns
    A tuple containing:
    * `X`: A tensor of shape [`point_count` x len(vars_model)] containing the sampled independent variables.
    * `y`: A tensor of shape [`point_count`] containing the evaluated equation values.
    """

    # of shape [N x D]
    X = torch.zeros(point_count, len(vars_model))
    X_dict = {}

    # set used variables to random values
    for idx, var in enumerate(vars_model):
        if var in vars_used:
            # sample uniformly and scale to the supported range
            X[:, idx] = torch.rand(point_count) * (max_support - min_support) + min_support

        X_dict[var] = X[:, idx]

    # evaluate equation
    y = sympy.lambdify(",".join(vars_model), eq)(**X_dict)

    return X, y
    
def sample_equation_from_config(
    eq: str, vars_used: list[str], point_count: int,
    model_cfg: omegaconf.DictConfig,
    eq_cfg: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper around `sample_equation` that uses the model and equation configuration to sample an equation.
    """

    return sample_equation(
        eq=eq,
        vars_used=vars_used,
        vars_model=eq_cfg["total_variables"],
        point_count=point_count,
        min_support=model_cfg.dataset_train.fun_support["min"],
        max_support=model_cfg.dataset_train.fun_support["max"],
    )

def generate_function(vars: list[str], identity_prob=0.5, nest_prob=0.3, max_depth=3) -> Tuple[str, list[str]]:
    """ 
    Generates a potentially nested function assuming a uniform function distribution over operators.
    This could, for example, be a combination of powers and trigonometric functions,
    but no addition, substraction, multiplication or division.

    # Args
    * `identity_prob`: The probability that the identity function is used (i.e., a naked variable).
    * `nest_prob`: In case the function is not an identity function, the probability of nesting a function.
    * `max_depth`: Maximum nesting depth.

    # Returns
    A tuple containing:
    * `equation`: The equation.
    * `vars_used`: The independent variables in the function.
    """

    # NOTE: functions that have incomplete domains are ignored (log, tan, etc.)
    funcs = ["abs", "cos", "exp", "sin"]
    
    var = np.random.choice(vars)

    if np.random.sample() < identity_prob:
        # return naked variable
        return var, [var]
    else:
        func = np.random.choice(funcs) + "("

        # nest function
        depth = 1
        while np.random.sample() < nest_prob and depth < max_depth:
            func += np.random.choice(funcs) + "("
            depth += 1

        # add variable
        func += var
        func += ")" * depth

        return func, [var]
    
def generate_simple_equation(vars: list[str], op_prob=1, decay=0.90, max_len=20) -> Tuple[str, list[str]]:
    """
    Generates an equation consisting only of addition, subtraction, multiplication and division.
    """

    funcs = ["+", "-", "*", "/"]

    # select initial variable
    var = np.random.choice(vars)
    func = var
    vars_used = { var }

    size = 1
    while np.random.sample() < op_prob and size < max_len:
        # select operator and variable
        op, var = np.random.choice(funcs), np.random.choice(vars)
        func += op + var
        
        vars_used.add(var)
        op_prob *= decay
        size += 1
    return func, list(vars_used)

def generate_dataset_pairs(
    strategy: Literal["sign-bias", "complexity-bias"], point_count: int, num_eq: int,
    model_cfg: omegaconf.DictConfig, eq_cfg: Dict,
    second_dataset_sample_rate: int=None    
) -> Dict[str, Union[torch.Tensor, list]]:
    """
    Generates dataset pairs based on a given strategy. Depending on the strategy, random equations pairs
    are generated, which are then sampled to form dataset pairs.
    
    # Args
    * `strategy`:
        1. If equal to `"sign-bias"`, equations are of the following form: `function_1(variable_1) ± function_2(variable_2)`.
        The first equation in the pair will apply the `+` operator and the second the `-` operator.
        2. If equal to `"complexity-bias"`, equations are sampled randomly. Each dataset in a pair samples the same
        equation, but the second dataset with a different sample rate determined by `second_dataset_sample_rate`.
    * `point_count`: The number of points per dataset.
    * `num_eq`: The number of equation pairs to generate.

    # Returns
    A dictionary containing:
    * `X0`: A tensor of sampled equation input variables of size [Ne x Np x D], where Ne denotes the number of equations,
        Np the number of points, and D the dimensionality.
    * `X1`: Analoguous to `X_dataset0`, but for the second dataset.
    * `y0`: A tensor of shape [Ne x Np] containing the evaluated equation values.
    * `y1`: Analoguous to `y_dataset0`, but for the second dataset.
    * `equations`: A list of size [Ne x 2] containing the generated equations.
    """
    output = { "equations": [] }

    if strategy == "sign-bias":
        output["X0"] = torch.empty((num_eq, point_count, len(eq_cfg["total_variables"])))
        output["y0"] = torch.empty((num_eq, point_count))
        output["X1"] = torch.empty((num_eq, point_count, len(eq_cfg["total_variables"])))
        output["y1"] = torch.empty((num_eq, point_count))

        for i in range(num_eq):
            # generate equation
            eq_part0, vars0 = generate_function(eq_cfg["total_variables"])
            eq_part1, vars1 = generate_function(eq_cfg["total_variables"])
            vars_used = vars0 + vars1

            eq_plus = eq_part0 + "+" + eq_part1
            eq_min = eq_part0 + "-" + eq_part1

            # sample equations
            output["X0"][i], output["y0"][i] = sample_equation_from_config(eq_plus, vars_used, point_count, model_cfg, eq_cfg)
            output["X1"][i], output["y1"][i] = sample_equation_from_config(eq_min, vars_used, point_count, model_cfg, eq_cfg)

            output["equations"].append((eq_plus, eq_min))

    elif strategy == "complexity-bias":
        assert second_dataset_sample_rate != None, f"second_dataset_sample_rate not set, but should be for strategy: {strategy}"

        output["X0"] = torch.empty((num_eq, point_count, len(eq_cfg["total_variables"])))
        output["y0"] = torch.empty((num_eq, point_count))

        # second dataset has fewer points
        output["X1"] = torch.empty((num_eq, point_count // second_dataset_sample_rate, len(eq_cfg["total_variables"])))
        output["y1"] = torch.empty((num_eq, point_count // second_dataset_sample_rate))

        for i in range(num_eq):
            # generate equation
            eq, vars_used = generate_simple_equation(eq_cfg["total_variables"])

            # sample equation
            Xe0, ye0 = sample_equation_from_config(eq, vars_used, point_count, model_cfg, eq_cfg)
            output["X0"][i], output["y0"][i] = Xe0, ye0
            # downsample second dataset
            output["X1"][i], output["y1"][i] = Xe0[::second_dataset_sample_rate], ye0[::second_dataset_sample_rate]

            output["equations"].append((eq, eq))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return output