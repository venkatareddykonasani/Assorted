# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def create_dataset(n, x_mean=10):
    """
    Create a dataset with n rows and two columns, "x" and "y",
    where "x" follows a normal distribution and "y" is correlated with "x"
    by the specified correlation value.

    Args:
        n (int): Number of rows in the dataset.
        correlation (float): Desired correlation between "x" and "y" (default: 0.9).

    Returns:
        pandas.DataFrame: A DataFrame containing the generated dataset.
    """
    # Generate x from a normal distribution
    x = np.random.normal(loc=x_mean, scale=1.0, size=n)

    # Generate y correlated with x
    noise = np.random.normal(loc=0.0, scale=1.0, size=n)
    y = 10*x + 25 + noise

    # Create a DataFrame with the generated data
    df = pd.DataFrame({'x': x, 'y': y})

    """## Creating data using the function"""

    #df = create_dataset(n=1000, x_mean=1)  # Create a dataset with 1000 rows
    #print(df.head())  # Print the first few rows
    #import matplotlib.pyplot as plt
    #plt.scatter(df.x, df.y)
    #plt.show()

    """# Regression Line"""


    return df

"""# MSE cost function Visualization"""

def visualization_3d(input_data):
  # define the vectorized MSE cost function
  def mse_cost(predictions, target):
      N = predictions.shape[0]
      diff = predictions.ravel() - target.ravel()
      cost = np.dot(diff, diff.T) / N
      return cost

  # define the prediction for a simple linear model
  def LinearModel(thetas, X):
      # normalize add bias term
      X = (X - X.mean()) / X.std()
      X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
      return np.dot(X, thetas)

  # initialize data structures
  #vis_df = df.sample(frac=0.1)
  vis_df= input_data
  y = vis_df.iloc[:, 1].to_numpy().reshape(-1, 1)
  X = vis_df.iloc[:, 0].to_numpy().reshape(-1, 1)

  # grid search over "all" possible theta values and compute cost
  start, end, step = -200, 200, 5
  thetas_0, thetas_1 = np.arange(start, end, step), np.arange(start, end, step)


  # loop over the all the parameter pairs and create a list of all possible pairs
  thetas_lst = []
  for theta_0 in thetas_0:
      for theta_1 in thetas_1:
          thetas_lst.append(np.array([theta_0, theta_1]).reshape(-1, 1))

  linear_cost_lst = []
  for thetas in thetas_lst:
      # get prediction from our model
      pred_linear = LinearModel(thetas, X)
      # keep track of the cast per parameter pairs
      linear_cost_lst.append(mse_cost(pred_linear, y))

  # arrange the costs back to a square matrix grid
  axis_length = len(np.arange(start, end, step))
  linear_cost_matrix = np.array(linear_cost_lst).reshape(axis_length, axis_length)

  import plotly.graph_objects as go

  # plot the surface plot with plotly's Surface
  fig = go.Figure(data=go.Surface(z=linear_cost_matrix,
                                  x=thetas_0,
                                  y=thetas_1))

  # add a countour plot
  fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))

  # annotate the plot
  fig.update_layout(title='Linear Model MSE Cost Surface',
                    scene=dict(
                      xaxis_title='X-Beta0',
                      yaxis_title='Y-Beta1',
                      zaxis_title='Z-Error Function'),
                    width=700, height=700)

  fig.show()
#visualization3d(input_data=df)
