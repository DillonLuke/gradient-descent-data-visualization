import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import animation

from functools import partial

def get_data(sample_size, b0, b1, x_mean, x_var, err_var):
    xs = np.random.normal(loc=x_mean, scale=np.sqrt(x_var), size=(sample_size, 1))
    errs = np.random.normal(loc=0, scale=np.sqrt(err_var), size=(sample_size, 1))
    
    X = np.hstack((np.ones(xs.shape), xs))
    betas = np.array([[b0], [b1]])
    
    y = (X @ betas) + errs 
    
    return X, y


def get_mse(b0, b1, X, y):
    betas = np.array([[b0], [b1]])
    
    preds = X @ betas
    
    squared_resid = (preds-y)**2
    
    mse = .5 * squared_resid.mean()
    
    return mse 


def gradient_descent(X, y, lr, tol):
    b0_cur, b1_cur, tol_cur = 0, 0, np.inf
    
    betas_list = [(b0_cur, b1_cur)]
    while tol_cur > tol:
        betas = np.array([[b0_cur], [b1_cur]])
        residuals = (X @ betas) - y 

        b0_grad = np.mean(residuals)
        b1_grad = np.mean(residuals * X[:, 1].reshape(-1, 1))
        
        tol_cur = np.sqrt((lr*b0_grad)**2 + (lr*b1_grad)**2)

        b0_cur -= (lr*b0_grad)
        b1_cur -= (lr*b1_grad)
        
        betas_list.append((b0_cur, b1_cur))
    
    b0s, b1s = list(zip(*betas_list))
    
    return b0s, b1s

    
def make_contour_plot(X, y, b0s, b1s, ax):
    x_range = np.linspace(min(b0s)*2, max(b0s)*2, 100)
    y_range = np.linspace(min(b1s)*2, max(b1s)*2, 100)
    
    xx, yy = np.meshgrid(x_range, y_range)
    get_mse_vec = np.vectorize(get_mse, excluded=["X", "y"])
    zz = get_mse_vec(b0=xx.flatten(), b1=yy.flatten(), X=X, y=y).reshape(xx.shape)
    
    ax.contourf(xx, yy, zz, levels=1000, cmap="RdYlBu")
    

def gradient_descent_frame(n, b0s, b1s, lines, title, step):
    lines.set_data(b0s[:n+1], b1s[:n+1])
    
    title.set_text(f"Iteration = {n*step}")
    
    return (lines, title)


def gradient_descent_animation(sample_size, b0, b1, x_mean, x_var, err_var, lr, tol):  
    X, y = get_data(sample_size, b0, b1, x_mean, x_var, err_var)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    title = ax.set_title("")
    lines, = ax.plot([], [], "k", lw=2)
    ax.set_xlabel("b0", size=12)
    ax.set_ylabel("b1", size=12)
    
    b0s, b1s = gradient_descent(X, y, lr, tol)
    
    make_contour_plot(X, y, b0s, b1s, ax)
    
    step = max(1, int(len(b0s)/100))
    gdf_partial = partial(gradient_descent_frame, b0s=b0s[::step], b1s=b1s[::step],
                          lines=lines, title=title, step=step)
    
    gd_animation = animation.FuncAnimation(fig, gdf_partial, frames=len(b0s[::step])-1,
                                            interval=50, blit=True)

    plt.close()
    
    return gd_animation
    