import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd

import math
import torch
import gpytorch
import numpy as np

import pickle


### Class definition
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# IMPORTING DATA
try:
    with open("pickles/model_info.pickle", "rb") as handle:
        gp_models = pickle.load(handle)

    with open("pickles/chain_info.pickle", "rb") as handle:
        daily_chains = pickle.load(handle)
except:
    pass

plt_day = {}
days = list(gp_models.keys())
for day_str in days:
    # holding each ttm info
    all_t = {}

    ttm = daily_chains[day_str]["calls"]["T"].unique()
    all_t["ttm"] = ttm

    # losses
    all_t["c_losses"] = gp_models[day_str]["call_losses"]
    all_t["p_losses"] = gp_models[day_str]["put_losses"]

    for t_to_exp in ttm:
        t_info = {}

        # getting training/test data
        c_tr = gp_models[day_str]["call_train"]
        c_tr = c_tr.loc[c_tr["T"] == t_to_exp]
        c_tst = gp_models[day_str]["call_test"]
        c_tst = c_tst.loc[c_tst["T"] == t_to_exp]

        p_tr = gp_models[day_str]["put_train"]
        p_tr = p_tr.loc[p_tr["T"] == t_to_exp]
        p_tst = gp_models[day_str]["put_test"]
        p_tst = p_tst.loc[p_tst["T"] == t_to_exp]

        # getting model/likelihood
        c_plt_model = torch.load("models/call_GP_" + day_str + ".pt")
        c_plt_model.eval()
        p_plt_model = torch.load("models/put_GP_" + day_str + ".pt")
        p_plt_model.eval()
        c_ll = gp_models[day_str]["call_likelihood"]
        c_ll.eval()
        p_ll = gp_models[day_str]["put_likelihood"]
        p_ll.eval()

        # sample data
        rng_mny = np.array(np.linspace(0.7, 1.3, 100)).astype(np.float32)
        samp_mny = (rng_mny - 0.7) / (1.3 - 0.7)
        t_arr = np.array([t_to_exp] * 100).astype(np.float32)
        samp_t = (t_arr - 20) / (365 - 20)

        sample = torch.tensor([samp_t, samp_mny]).T  # .reshape((1000, 2))

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            c_plt = c_ll(c_plt_model(sample))
            p_plt = p_ll(p_plt_model(sample))

        t_info["ctrain"] = c_tr
        t_info["ctest"] = c_tst
        t_info["ptrain"] = p_tr
        t_info["ptest"] = p_tst

        # undo exponentiation
        c_lower, c_upper = c_plt.confidence_region()
        c_lower = math.e ** c_lower.numpy()
        c_upper = math.e ** c_upper.numpy()
        c_preds = math.e ** c_plt.mean.numpy()

        iv = {}
        iv["c_lower"] = c_lower
        iv["c_upper"] = c_upper
        iv["c_mean"] = c_preds
        iv["mny"] = rng_mny

        p_lower, p_upper = p_plt.confidence_region()
        p_lower = math.e ** p_lower.numpy()
        p_upper = math.e ** p_upper.numpy()
        p_preds = math.e ** p_plt.mean.numpy()

        iv["p_lower"] = p_lower
        iv["p_upper"] = p_upper
        iv["p_mean"] = p_preds

        t_info["iv"] = pd.DataFrame(iv)
        all_t[t_to_exp] = t_info

    plt_day[day_str] = all_t


# Initialize Dash app
app = dash.Dash()

# App layout
app.layout = html.Div(
    [
        dcc.Dropdown(
            id="day_dropdown",
            options=[{"label": i, "value": i} for i in days],
            value=days[0],
        ),
        dcc.Slider(
            id="ttm_slider",
            # Initial range, will be updated based on day selection
            min=min(plt_day["2023-10-02"]["ttm"]),
            max=max(plt_day["2023-10-02"]["ttm"]),
            step=None,
            marks={str(t): str(t) for t in plt_day["2023-10-02"]["ttm"]},
            value=min(plt_day["2023-10-02"]["ttm"]),
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        dcc.Graph(id="call_graph"),
        dcc.Graph(id="put_graph"),
        dcc.Graph(id="call_surface", style={"height": "80vh", "width": "80vh"}),
        dcc.Graph(id="put_surface", style={"height": "80vh", "width": "80vh"}),
        dcc.Graph(id="losses"),
    ]
)


# Callback to update the strike slider based on selected day
@app.callback(
    Output("ttm_slider", "min"),
    Output("ttm_slider", "max"),
    Output("ttm_slider", "step"),
    Output("ttm_slider", "marks"),
    Output("ttm_slider", "value"),
    [Input("day_dropdown", "value")],
)
def update_slider(selected_day):
    ttms = plt_day[selected_day]["ttm"]
    # print(ttms)
    marks = {str(i): str(i) + " days" for i in ttms}
    return min(ttms), max(ttms), None, marks, min(ttms)


# Callback to update calls
@app.callback(
    Output("call_graph", "figure"),
    [Input("day_dropdown", "value"), Input("ttm_slider", "value")],
)
def update_graph(selected_day, selected_time):
    if selected_time is None or selected_day is None:
        raise dash.exceptions.PreventUpdate

    df = plt_day[selected_day][selected_time]

    fig = go.Figure()
    # Add traces for training and testing data

    # Add trace for mean function
    fig.add_trace(
        go.Scatter(
            x=df["iv"]["mny"],
            y=df["iv"]["c_mean"],
            mode="lines",
            line=dict(color="red"),
            name="mean function",
        )
    )

    # Add shaded area for confidence interval
    fig.add_trace(
        go.Scatter(
            x=df["iv"]["mny"],
            y=df["iv"]["c_lower"],
            fill=None,
            mode="lines",
            line=dict(color="lightgrey"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["iv"]["mny"],
            y=df["iv"]["c_upper"],
            fill="tonexty",
            mode="lines",
            line=dict(color="lightgrey"),
            name="95% confidence",
        )
    )
    # testing/training points
    fig.add_trace(
        go.Scatter(
            x=df["ctrain"]["mny"],
            y=df["ctrain"]["iv"],
            mode="markers",
            marker=dict(color="black"),
            name="training points",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["ctest"]["mny"],
            y=df["ctest"]["iv"],
            mode="markers",
            marker=dict(color="blue"),
            name="testing points",
        )
    )

    fig.update_layout(
        title="Call option IV surface for Time to Expiration of "
        + str(selected_time)
        + " days.",
        xaxis_title="Moneyness",
        yaxis_title="Implied Volatility",
    )

    return fig


# Callback to update puts
@app.callback(
    Output("put_graph", "figure"),
    [Input("day_dropdown", "value"), Input("ttm_slider", "value")],
)
def update_graph(selected_day, selected_time):
    if selected_time is None or selected_day is None:
        raise dash.exceptions.PreventUpdate

    df = plt_day[selected_day][selected_time]

    fig = go.Figure()
    # Add traces for training and testing data

    # Add trace for mean function
    fig.add_trace(
        go.Scatter(
            x=df["iv"]["mny"],
            y=df["iv"]["p_mean"],
            mode="lines",
            line=dict(color="red"),
            name="mean function",
        )
    )

    # Add shaded area for confidence interval
    fig.add_trace(
        go.Scatter(
            x=df["iv"]["mny"],
            y=df["iv"]["p_lower"],
            fill=None,
            mode="lines",
            line=dict(color="lightgrey"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["iv"]["mny"],
            y=df["iv"]["p_upper"],
            fill="tonexty",
            mode="lines",
            line=dict(color="lightgrey"),
            name="95% confidence",
        )
    )
    # testing/training points
    fig.add_trace(
        go.Scatter(
            x=df["ptrain"]["mny"],
            y=df["ptrain"]["iv"],
            mode="markers",
            marker=dict(color="black"),
            name="training points",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["ptest"]["mny"],
            y=df["ptest"]["iv"],
            mode="markers",
            marker=dict(color="blue"),
            name="testing points",
        )
    )

    fig.update_layout(
        title="Put option IV surface for Time to Expiration of "
        + str(selected_time)
        + " days.",
        xaxis_title="Moneyness",
        yaxis_title="Implied Volatility",
    )

    return fig


# Callback to update puts
@app.callback(
    Output("call_surface", "figure"),
    [Input("day_dropdown", "value")],
)
def plot_3d(selected_day):
    day = plt_day[selected_day]
    ttms = day["ttm"]
    mny = day[ttms[0]]["iv"]["mny"]

    # Initialize a 2D array for z values
    surface = np.zeros((len(mny), len(ttms)))

    # Populate z_values
    for j, t in enumerate(ttms):
        df = day[t]["iv"]
        for i in range(len(mny)):
            surface[i, j] = df["c_mean"][i]

    # surface = np.array(surface)
    fig = go.Figure(data=[go.Surface(x=mny, y=ttms, z=surface.T)])

    # Update layout for a better view
    fig.update_layout(
        title="Call Volatility Surface",
        width=800,  # Width of the plot
        height=800,  # Height of the plot
        scene=dict(
            xaxis_title="Moneyness",
            yaxis_title="Time to Expiration",
            zaxis_title="Implied Volatility",
        ),
    )

    return fig


# Callback to update puts
@app.callback(
    Output("put_surface", "figure"),
    [Input("day_dropdown", "value")],
)
def plot_3d(selected_day):
    day = plt_day[selected_day]
    ttms = day["ttm"]
    mny = day[ttms[0]]["iv"]["mny"]

    # Initialize a 2D array for z values
    surface = np.zeros((len(mny), len(ttms)))

    # Populate z_values
    for j, t in enumerate(ttms):
        df = day[t]["iv"]
        for i in range(len(mny)):
            surface[i, j] = df["p_mean"][i]

    # surface = np.array(surface)
    fig = go.Figure(data=[go.Surface(x=mny, y=ttms, z=surface.T)])

    # Update layout for a better view
    fig.update_layout(
        title="Put Volatility Surface",
        width=800,  # Width of the plot
        height=800,  # Height of the plot
        scene=dict(
            xaxis_title="Moneyness",
            yaxis_title="Time to Expiration",
            zaxis_title="Implied Volatility",
        ),
    )

    return fig


# Callback to update calls
@app.callback(
    Output("losses", "figure"),
    [Input("day_dropdown", "value")],
)
def update_graph(selected_day):
    if selected_day is None:
        raise dash.exceptions.PreventUpdate

    df = plt_day[selected_day]

    fig = go.Figure()
    # Add traces for training and testing data

    # Add trace for mean function
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, len(df["c_losses"]), 1),
            y=df["c_losses"],
            mode="lines",
            line=dict(color="red"),
            name="Call loss",
        )
    )

    # Add shaded area for confidence interval
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, len(df["p_losses"]), 1),
            y=df["p_losses"],
            mode="lines",
            line=dict(color="blue"),
            name="Put loss",
        )
    )

    fig.update_layout(
        title="Call and Put Model Losses",
        xaxis_title="Training Iterations",
        yaxis_title="Marginal Log Likelihood",
    )

    return fig


# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
    print(plt_day["2023-10-02"]["ttm"])
# print(plt_day.keys())
#
# print(plt_day["2023-10-02"][25].keys())
