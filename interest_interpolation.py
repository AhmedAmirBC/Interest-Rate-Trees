#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def yield_func(t, p):
    chunk_1 = (p["a2"] + p["a3"]) * (p["b"]/t) * (1 - np.exp(-t/p["b"]))
    chunk_2 = p["a3"] * np.exp(-t/p["b"])
    return p["a1"] + chunk_1 - chunk_2


def plot_term(p, mdata):
    x = [i for i in range(1, 31)]
    y = [yield_func(i, p) for i in x]
    plt.plot(x, y, label="Estimated Term Structure")
    plt.plot(mdata.keys(), mdata.values(), label="Actual Term Structure")
    plt.show()


def make_params(a1, a2, a3, b):
    params = {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "b": b
    }
    return params



def make_data(terms, yields):
    data =  dict(zip(terms, yields))
    return data


def test_params(p, mdata, plot=False):
    "Finds Errors for the parameters, which is the sum of squares of differences between actual and estimated yields/"
    x = [i for i in range(1, 31)]
    y = [yield_func(i, p) for i in x]
    
    if plot:
        plt.plot(x, y, label="estimated")
        plt.plot(data.keys(), mdata.values(), label="actual")
        plt.title("Yield Curve, Estimated vs Actual")
        plt.xlabel("maturity (years)")
        plt.ylabel("interest rate (%)")
        plt.show()
    
    errors = []
    assert(type(mdata) == dict)
    for data in mdata:
        yield_i = y[data - 1]
        estimate_yield_i = mdata.get(data)
        error = (yield_i - estimate_yield_i)**2
        errors.append(error)
    return sum(errors)


def error_derivative(var, p, data, verbose=False):
    "Estimates the derivative of the given variable"
    upper_params = make_params(p["a1"], p["a2"], p["a3"], p["b"])
    lower_params = make_params(p["a1"], p["a2"], p["a3"], p["b"])
    
    upper_params[var] += 0.01
    lower_params[var] -= 0.01
    
    upper_error = test_params(upper_params, data, False)
    lower_error = test_params(lower_params, data, False)
    
    if verbose:
        print("Upper Error: ", upper_error)
        print("Lower Error: ", lower_error)
    
    return (upper_error - lower_error)/(2 * 0.01) # the derivative


def error_gradient(p, data, verbose=False):
    """returns a gradient of derivatives."""
    assert type(data) == dict
    variables = list(p.keys())
    derivatives = []
    for var in variables:
        derivative = error_derivative(var, p, data, verbose)
        derivatives.append(derivative)
    return dict(zip(variables, derivatives))


def change_params(p, grad):
    for var in p:
        p[var] -= grad[var] * 0.02
    return p

def gradient_descent_trial(p, data, speed=0.1):
    assert type(data) == dict
    grad = error_gradient(p, data)
    p = change_params(p, grad)
    return p

def fit_parameters(terms, yields):
    """Main Algorithm.
    
    Just insert the term (list) and corresponding yields to maturity (list) 
    and get back a set of parameters which construct a continuous term structure.
    """
    data = make_data(terms, yields)
    params = make_params(0.5, 0.5, 0.5, 1)
    
    assert(type(data) == dict)
    
    speed = 0.02
    errors = []
    while test_params(params, data) > 0.0039:
        if len(errors) > 10 and (abs(errors[-1] - errors[0])) < 0.01:
            speed *= 20
            params = gradient_descent_trial(params, data, speed)
        else:
            params = gradient_descent_trial(params, data, speed)
        error = test_params(params, data)
        errors.append(error)
    print("error: ", error)
    return params


if __name__ == "__main__":
    data = make_data([1, 2, 3, 5, 7, 10, 20, 30], [2.03, 1.9,  1.87, 1.91, 2.03, 2.15, 2.42, 2.62])
    params = fit_parameters(data.keys(), data.values())
    plot_term(params, data)