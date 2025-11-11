# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta

    from sklearn.linear_model import RidgeCV

    return RidgeCV, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""# Forecast Tool""")
    return


@app.cell
def _(mo, pd):
    def params():
        # Policy / simulation params
        sigma = 0.0000005 # inflation change step per round
        P_star = 0.5 # target participation rate P^*
        gamma_min = 0.0001 # lower bound of inflation per round
        gamma_max = 0.01 # upper bound of inflation per round

        # Simulation
        horizon_days = 180
        n_sims = 20
        random_seed = 42

        # Risk band for participation (D0 = [Plow, Phigh])
        Plow = 0.40
        Phigh = 0.60

        # Admissibility thresholds (from framework)
        T_star = 10 # expected number of days outside D0 allowed over horizon
        Ttail = 20 # tail threshold: unacceptable if time-outside > Ttail
        eps_tail = 0.05 # allowed probability of exceeding Ttail across sims
        gamma_star = 0.25 # target emission rate
        yield_star = 0.4 # target yield rate
    
        return dict(sigma=sigma, P_star=P_star, gamma_min=gamma_min, gamma_max=gamma_max,
                    horizon_days=horizon_days, n_sims=n_sims,
                    random_seed=random_seed, Plow=Plow, Phigh=Phigh,
                    T_star=T_star, Ttail=Ttail, eps_tail=eps_tail)


    # ------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------
    path = "/Users/sazisbekuu/Downloads/ShtukaResearch/DATA.csv"    # adjust path if needed
    df_raw = pd.read_csv(path)

    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

    # Fix the inflation calculation:
    df_raw["inflation_per_round"] = df_raw["inflation"]/1e9  # inflation per round
    df_raw["annual_inflation_rate"] = (1 + df_raw["inflation_per_round"]) ** 417 - 1 # annualized issuance rate

    df = df_raw.dropna(subset=["date"]).set_index("date").sort_index()

    # Identify key columns
    p_col = "participation-rate"
    g_col = "annual_inflation_rate"

    # Convert numerics
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[p_col, g_col])
    mo.ui.data_explorer(df)

    return df, params


@app.cell
def _(mo):
    mo.md(r"""# Data Preparation""")
    return


@app.cell
def _(df, mo, np, pd, plt):
    def prepare_data(df, target_col, features, cutoff_date):
        # We'll construct the regression to predict Y_{t+1} from [1, Y_t, gamma_t, x_t]
        df2 = df.copy()

        # Find rows with any NaN
        rows_with_nan = df2[df2.isnull().any(axis=1)]

        # Fill NaN in those columns with their respective mean
        df2['fear-greed-index'] = df2['fear-greed-index'].fillna(df2['fear-greed-index'].mean())
    
        df2.rename(columns={'participation-rate': 'P'}, inplace=True)
        df2.rename(columns={'annual_inflation_rate': 'I'}, inplace=True)
        df2['Y'] = np.log(df2['P']/(1-df2['P']))
        df2['logP'] = np.log(df2['P'])
        # target y = Y_{t+1}
        df2['Y_next'] = df2['Y'].shift(-1)
        df2['logP_next'] = df2['logP'].shift(-1)
        df2['P_next'] = df2['P'].shift(-1)
        # drop last row with NaN target
        df2 = df2.iloc[:-1]

        # exogenous features: 
        exog_cols = [c for c in df2.columns if c in features]
        # design matrix columns: intercept, Y_t, gamma_t, exog...
        X = pd.DataFrame(index=df2.index)
        X['intercept'] = 1.0
        if target_col == 'logit':
            X['Y_t'] = df2['Y']
            y = df2['Y_next'].values
        elif target_col == 'log':
            X['logP_t'] = df2['logP']
            y = df2['logP_next'].values
        else:
            X['P_t'] = df2['P']
            y = df2['P_next'].values
        
        X['I_t'] = df2['I']
        for c in exog_cols:
            if c not in ['intercept', 'Y_t', 'logP_t', 'P_t','I_t']:
                X[c] = df2[c]        

        # Split to Training Set
        cutoff = pd.to_datetime(cutoff_date)
        X_train = X.loc[:cutoff]
        X_test = X.loc[cutoff:]
        cutoff_loc = X.index.get_loc(cutoff)
        y_train = y[:cutoff_loc + 1]
        y_test = y[cutoff_loc + 1:]
    
        return X_train, y_train, X_test, y_test

    # Date picker UI
    date_picker = mo.ui.date(label="Select cutoff date", value="2024-10-31")

    # Reactive cell: plot and split
    def plot_and_split(cutoff_date):
        cutoff = pd.to_datetime(cutoff_date)
        train_data = df[:cutoff]
        test_data = df[cutoff:]

        # Plot DataFrame
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df.index, df["participation-rate"], label="Participation Rate", color="blue")
        ax.axvline(cutoff, color="red", linestyle="--", label="Cutoff Date")
        ax.set_title("Data over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("P")
        ax.legend()

        return mo.vstack([
            fig,
            mo.md(f"**Cutoff Date:** {cutoff_date}"),
            mo.md(f"Training Size: {len(train_data)} days"),
            mo.md(f"Test Size: {len(test_data)} days")
        ])


    '''feat_cols = ["btc_price_usd", "eth_price_usd", "fear-greed-index"]

    X, y = prepare_data(df, 'logit', feat_cols)

    print(X.columns)
    plt.figure()
    plt.plot(X['Y_t'])
    plt.plot(X.index,y)
    plt.show()'''

    return date_picker, plot_and_split, prepare_data


@app.cell
def _(date_picker, mo, plot_and_split):
    # Display UI and plot
    mo.hstack([
        date_picker,
        plot_and_split(date_picker.value)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""# Estimation""")
    return


@app.cell
def _(RidgeCV, df, np):
    def fit_linear(X, y):
        dP = df['P'].diff().dropna()
        P_lag = df['P'].shift(1).dropna()
        # regress dP on (P_tau - P_lag)
        X = (0.5 - P_lag).loc[dP.index].values.reshape(-1,1)  # using 0.5 as example
        beta = np.linalg.lstsq(X, dP.values, rcond=None)[0][0]
        sigma_eps = np.std(dP - beta * X.ravel())
        return dict(beta=float(beta), sigma_eps=float(sigma_eps))


    def fit_ridge(X, y, exog_cols):
        model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 2, 4, 10, 20, 50], cv=5)
        model.fit(X, y)
        best_lambda = model.alpha_


        n_features = X.shape[1]
        # ridge closed form: theta = (X^T X + lambda * I)^{-1} X^T y
        I = np.eye(n_features)
        I[0,0] = 0 # don't regularize intercept
    
        beta = np.linalg.inv(X.T @ X + best_lambda * I) @ X.T @ y

        # split coefficients
        coef = dict()
        for i, c in enumerate(exog_cols):
            coef[c] = float(beta[i])


        # residual variance
        y_pred = X @ beta
        resid = y - y_pred
        sigma_eps = float(np.std(resid, ddof=1))

        return coef, beta, sigma_eps

    '''coef, beta, eps = fit_ridge(X.values, y, X.columns.to_list())

    print(coef)
    print(eps)
    print(list(coef.keys()))'''
    return (fit_ridge,)


@app.cell
def _(mo):
    mo.md(r"""# Simulation""")
    return


@app.cell
def _(
    date_picker,
    df,
    fit_ridge,
    mo,
    np,
    params,
    plt,
    prepare_data,
    radio_horizon,
    radio_paths,
    radio_sampling,
    slider_gamma_max,
    slider_gamma_min,
    slider_sigma,
):
    def sample_exog(series, n_paths, horizon, method="bootstrap", block_size=None, random_state=None):
        """
        Simulate future paths for a single exogenous variable using block bootstrap or AR(1).
    
        Parameters:
        -----------
        series : pd.Series
            Historical data for the exogenous variable.
        n_paths : int
            Number of simulated paths.
        horizon : int
            Forecast horizon (number of future steps).
        method : str
            "bootstrap" for historical block bootstrapping, "ar1" for AR(1) simulation.
        block_size : int or None
            Block size for bootstrap (required if method="bootstrap").
        random_state : int or None
            Seed for reproducibility.
    
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, horizon) with simulated future samples.
        """
        if random_state is not None:
            np.random.seed(random_state)
    
        data = series.values
        n_obs = len(data)
        samples = np.zeros((n_paths, horizon))
    
        if method == "bootstrap":
            if block_size is None:
                raise ValueError("block_size must be provided for bootstrap method.")
        
            num_blocks = int(np.ceil(horizon / block_size))
        
            for path in range(n_paths):
                blocks = []
                for _ in range(num_blocks):
                    start = np.random.randint(0, n_obs - block_size + 1)
                    block = data[start:start + block_size]
                    blocks.append(block)
                sample = np.concatenate(blocks)[:horizon]
                samples[path] = sample
    
        elif method == "ar1":
            # Fit AR(1): y_t = phi * y_{t-1} + epsilon
            y_lag = data[:-1]
            y_curr = data[1:]
            phi = np.dot(y_lag, y_curr) / np.dot(y_lag, y_lag)
            sigma = np.std(y_curr - phi * y_lag)
            last_val = data[-1]
        
            for path in range(n_paths):
                sim = [last_val]
                for _ in range(horizon - 1):
                    next_val = phi * sim[-1] + np.random.normal(0, sigma)
                    sim.append(next_val)
                samples[path] = sim
    
        else:
            raise ValueError("method must be either 'bootstrap' or 'ar1'.")
    
        return samples


    def simulate(df, target_col, exog_variables, params):
        np.random.seed(params['random_seed'])
        n = params['n_sims']
        H = params['horizon_days']
        sigma = params['sigma']
        P_star = params['P_star']
        gamma_min = params['gamma_min']
        gamma_max = params['gamma_max']

        X, y, X_test, y_test = prepare_data(df, target_col, exog_variables, date_picker.value)
        # unpack fit
        coef, beta, ridge_eps = fit_ridge(X.values, y, X.columns.to_list())


        # initial states
        if target_col == 'logit':
            P0 = X['Y_t'].iloc[-1]
            beta_P = coef['Y_t']
        elif target_col == 'log':
            P0 = X['logP_t'].iloc[-1]
            beta_P = coef['logP_t']
        else: 
            P0 = X['P_t'].iloc[-1]
            beta_P = coef['P_t']
        
        I0 = X['I_t'].iloc[-1]


        # For exogenous regressors we'll bootstrap (with replacement) historical rows
        X_future = np.zeros((n, H, len(exog_variables)))
        if radio_sampling.value == 'bootstrap':
            for i, c in enumerate(exog_variables):
                X_future[:, :, i] = sample_exog(X[c], n, H, method='bootstrap', block_size=7, random_state=42)
        else:
            for i, c in enumerate(exog_variables):
                X_future[:, :, i] = sample_exog(X[c], n, H, method='ar1', block_size=7, random_state=42)


        # storage
        P_paths = np.zeros((n, H+1))
        I_paths = np.zeros((n, H+1))
        P_paths[:,0] = P0
        I_paths[:,0] = I0


        # precompute which indices in theta correspond to which features
        # cols order: intercept, Y_t, gamma_t, (exog...)
        beta_intercept = coef['intercept']
        beta_I = coef['I_t']
        beta_exog = np.array([coef[c] for c in list(coef.keys()) if c not in ['intercept','Y_t', 'logP_t', 'P_t','I_t']]).reshape(-1,1)

        for t in range(H):
            # sample exogenous for this step for all sims
            exog_vals = X_future[:, t, :]

            # compute Y_t, gamma_t arrays
            P_t = P_paths[:, t]
            I_t = I_paths[:, t]

            
            # build X_t: shape (n, n_features)
            '''X_t = np.ones((n, len(exog_variables)))
            X_t[:,Y_idx] = Y_t
            X_t[:,gamma_idx] = gamma_t
            for j, ex_i in enumerate(exog_idxs):
                X_t[:, ex_i] = exog_vals[:, j]'''

            # dynamics for Y_{t+1}
            Y_mean = beta_intercept + beta_P * P_t + beta_I * I_t + (exog_vals @ beta_exog).ravel()
        
            #Y_mean = X_t @ theta_vec
            eps = np.random.randn(n) * ridge_eps
            Y_next = Y_mean + eps
        
            if target_col == 'logit':
                P_curr = 1/(1+np.exp(-P_t))
            elif target_col == 'log':
                P_curr = np.exp(P_t)
            else:
                P_curr = P_t

            # apply protocol policy to update gamma: gamma_{t+1} = clip(gamma_t + sigma * sign(P_star - P_t), [gamma_min, gamma_max])
            control = sigma * np.sign(P_star - P_curr)
            I_t_per_round = (I_t + 1) ** (1/417) - 1
            I_next_per_round = np.clip(I_t_per_round + control, gamma_min, gamma_max)
            I_next = (1 + I_next_per_round) ** 417 - 1 # adjusted annualized issuance rate
            #I_next = np.clip(I_t + control, gamma_min, gamma_max)

            P_paths[:,t+1] = Y_next
            I_paths[:,t+1] = I_next

        if target_col == 'logit':
            P_paths = 1/(1+np.exp(-P_paths))
            y_test = 1/(1+np.exp(-y_test))
        elif target_col == 'log':
            P_paths = np.exp(P_paths)
            y_test = np.exp(y_test)
    
        return P_paths, I_paths, X_future, X_test, y_test, coef


    def simulation_plots():
        exog_cols = ["btc_price_usd", "eth_price_usd", "fear-greed-index"]
        parameters = params()

        # Adjusted Parameters:
        parameters['gamma_max'] = slider_gamma_max.value
        parameters['gamma_min'] = slider_gamma_min.value
        parameters['sigma'] = slider_sigma.value * 1e-9   # adjust to the correct value (ppb)
        parameters['n_sims'] = int(radio_paths.value)
        parameters['horizon_days'] = int(radio_horizon.value)
    
        P_paths, I_paths, X_paths, X_test, y_test, optimal_beta = simulate(df, 'logit', exog_cols, parameters)
    
        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
        horizon = parameters['horizon_days']
        p10 = np.percentile(P_paths, 10, axis=0)
        p25 = np.percentile(P_paths, 25, axis=0)
        p50 = np.percentile(P_paths, 50, axis=0)
        p75 = np.percentile(P_paths, 75, axis=0)
        p90 = np.percentile(P_paths, 90, axis=0)
    
        ax1.fill_between(range(0, horizon + 1), p10, p90, color='skyblue', alpha=0.4, label='90% interval')
        ax1.fill_between(range(0, horizon + 1), p25, p75, color='dodgerblue', alpha=0.6, label='50% interval')
        ax1.plot(range(0, horizon + 1), p50, color='blue', linewidth=2, label='Median')
        ax1.plot(range(0, len(y_test)), y_test, color='red', linewidth=2, label='True Value')
    
        ax1.set_ylabel('Participation Rate')
        ax1.set_title('Forecast Fan Charts')
    
        # Plot on second axis
        p10 = np.percentile(I_paths, 10, axis=0)
        p25 = np.percentile(I_paths, 25, axis=0)
        p50 = np.percentile(I_paths, 50, axis=0)
        p75 = np.percentile(I_paths, 75, axis=0)
        p90 = np.percentile(I_paths, 90, axis=0)
    
        ax2.fill_between(range(0, horizon + 1), p10, p90, color='skyblue', alpha=0.4, label='90% interval')
        ax2.fill_between(range(0, horizon + 1), p25, p75, color='dodgerblue', alpha=0.6, label='50% interval')
        ax2.plot(range(0, horizon + 1), p50, color='blue', linewidth=2, label='Median')
        ax2.plot(range(0, len(X_test['I_t'])), X_test['I_t'], color='red', linewidth=2, label='True Value')
    
        ax2.set_ylabel('Issuance Rate')
        ax2.set_xlabel('Horizon Days')
    
        plt.legend()
        plt.tight_layout()
        plt.show()

        return mo.vstack([mo.md(f"$\\hat{{\\beta}}$: {optimal_beta}"), fig ])

    return simulate, simulation_plots


@app.cell
def _(mo):

    # Create sliders for different parameters
    slider_gamma_max = mo.ui.slider(start=0.00001, stop=0.002, step=0.00002, value=0.0007, label="Issuance Max")
    slider_gamma_min = mo.ui.slider(start=0.000001, stop=0.001, step=0.00002, value=0.0006, label="Issuance Min")
    slider_sigma = mo.ui.slider(start=100, stop=10000, step=200, value=500, label="Inflation Change (ppb)")
    radio_paths = mo.ui.radio(
        options=['5', '20', '200', '1000'],
        value='20',  # default selection
        label="Number of simulations"
    )
    radio_horizon = mo.ui.radio(
        options=['30', '90', '180'],
        value='180',  # default selection
        label="Horizon Days"
    )
    radio_sampling = mo.ui.radio(
        options=['bootstrap', 'AR1'],
        value='bootstrap',  # default selection
        label="Sampling of Exogeneous Variables"
    )


    return (
        radio_horizon,
        radio_paths,
        radio_sampling,
        slider_gamma_max,
        slider_gamma_min,
        slider_sigma,
    )


@app.cell
def _(
    mo,
    radio_horizon,
    radio_paths,
    radio_sampling,
    simulation_plots,
    slider_gamma_max,
    slider_gamma_min,
    slider_sigma,
):
    # Display parameters and their current values
    mo.vstack([
        mo.hstack([
            mo.vstack([slider_gamma_max, slider_gamma_min, slider_sigma]),
            mo.vstack([slider_gamma_max.value, slider_gamma_min.value, slider_sigma.value])
                ]),
        mo.hstack([
            radio_paths, radio_horizon, radio_sampling 
                ]),
        simulation_plots()
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""# Risk Assessment""")
    return


@app.cell
def _(
    df,
    mo,
    np,
    params,
    radio_horizon,
    radio_paths,
    simulate,
    slider_Phigh,
    slider_Plow,
    slider_Teps,
    slider_Tstar,
    slider_Ttail,
    slider_gamma_max,
    slider_gamma_min,
    slider_gamma_star,
    slider_sigma,
    slider_yield_star,
):
    def risk_admissibility():
        exog_cols = ["btc_price_usd", "eth_price_usd", "fear-greed-index"]
        parameters = params()

        # Adjusted Parameters:
        parameters['gamma_max'] = slider_gamma_max.value
        parameters['gamma_min'] = slider_gamma_min.value
        parameters['sigma'] = slider_sigma.value
        parameters['n_sims'] = int(radio_paths.value)
        parameters['horizon_days'] = int(radio_horizon.value)
    
        P_paths, I_paths, X_paths, X_test, y_test, optimal_beta = simulate(df, 'logit', exog_cols, parameters)
    
        P = P_paths
        H = parameters['horizon_days']
        Plow = slider_Plow.value
        Phigh = slider_Phigh.value
        T_star = slider_Tstar.value
        Ttail = slider_Ttail.value
        eps_tail = slider_Teps.value
        gamma_star = slider_gamma_star.value
        yield_star = slider_yield_star.value

        # compute time outside D0 for each sim (count of days where P not in [Plow,Phigh])
        outside = ((P < Plow) | (P > Phigh)).sum(axis=1) # includes t=0..H

        expected_outside = float(np.mean(outside))
        prob_exceed_tail = float((outside > Ttail).mean())

        admissible = (expected_outside <= T_star) and (prob_exceed_tail <= eps_tail)

        # Emission and Yield Rate target acceptance
        ET = I_paths[:,-1].mean()
        YT = I_paths/P_paths
        YT = YT[:,-1].mean()

        admissible = (admissible) and (ET <= gamma_star) and (YT <= yield_star)

        # also compute percentiles for plotting
        q10 = np.percentile(P, 10, axis=0)
        q50 = np.percentile(P, 50, axis=0)
        q90 = np.percentile(P, 90, axis=0)

        result = dict(expected_outside=expected_outside, prob_exceed_tail=prob_exceed_tail,
        admissible=bool(admissible), q10=q10, q50=q50, q90=q90)
        # human readable summary
        print(f"Expected days outside D0 over {H} days: {expected_outside:.2f} (threshold T*={T_star})")
        print(f"Probability time-outside > {Ttail}: {prob_exceed_tail:.3f} (allowed eps={eps_tail})")
        print(f"Emission rate: {ET:.3f} (acceptance target={gamma_star})")
        print(f"Yield: {YT:.3f} (acceptance target={yield_star})")
        print('RISK-ADMISSIBLE:' , '✅ YES' if admissible else '❌ NO')

        #return result
        return mo.vstack([mo.md(f"Expected days outside D0 over {H} days: {expected_outside:.2f} (threshold T*={T_star})"),
                          mo.md(f"Probability time-outside > {Ttail}: {prob_exceed_tail:.3f} (allowed eps={eps_tail})"),
                          mo.md(f"Emission rate: {ET:.3f} (acceptance target={gamma_star})"),
                          mo.md(f"Yield: {YT:.3f} (acceptance target={yield_star})"),
                          mo.md(f"RISK-ADMISSIBLE: {'✅ YES' if admissible else '❌ NO'}")
                         ])

    #result_risk = risk_admissibility()

    return (risk_admissibility,)


@app.cell
def _(mo):
    # UI for risk admissibility parameters
    slider_Plow = mo.ui.slider(start=0.0, stop=0.5, step=0.01, value=0.4, label="P_low")
    slider_Phigh = mo.ui.slider(start=0.5, stop=1.0, step=0.01, value=0.6, label="P_high")
    slider_Tstar = mo.ui.slider(start=1, stop=100, step=1, value=10, label="T_star")
    slider_Ttail = mo.ui.slider(start=1, stop=100, step=1, value=20, label="T_tail")
    slider_Teps = mo.ui.slider(start=0.01, stop=0.3, step=0.01, value=0.05, label="T_eps")
    slider_gamma_star = mo.ui.slider(start=0.05, stop=1.0, step=0.01, value=0.25, label="gamma_star")
    slider_yield_star = mo.ui.slider(start=0.1, stop=1.0, step=0.01, value=0.4, label="yield")
    return (
        slider_Phigh,
        slider_Plow,
        slider_Teps,
        slider_Tstar,
        slider_Ttail,
        slider_gamma_star,
        slider_yield_star,
    )


@app.cell
def _(
    mo,
    risk_admissibility,
    slider_Phigh,
    slider_Plow,
    slider_Teps,
    slider_Tstar,
    slider_Ttail,
    slider_gamma_star,
    slider_yield_star,
):
    mo.vstack([
        mo.md("### Risk-Admissibility Parameters"),
        mo.hstack([mo.vstack([slider_Plow, slider_Phigh, slider_Tstar, slider_Ttail, slider_Teps, slider_gamma_star, slider_yield_star]), 
                  mo.vstack([mo.md(f"P_low: **{slider_Plow.value}**"), mo.md(f"P_high: **{slider_Phigh.value}**"), mo.md(f"T_star: **{slider_Tstar.value}**"), mo.md(f"T_tail: **{slider_Ttail.value}**"), mo.md(f"eps_Tail: **{slider_Teps.value}**"), mo.md(f"gamma_star: **{slider_gamma_star.value}**"), mo.md(f"yield_star: **{slider_yield_star.value}**")]) ]),
        risk_admissibility()
    ])


    return


@app.cell
def _(np, params, plt):
    def diagnostics(dates, P_paths, risk_admissibility):
        q10 = risk_admissibility['q10']
        q50 = risk_admissibility['q50']
        q90 = risk_admissibility['q90']

        plt.figure(figsize=(10,4))
        plt.fill_between(dates, q10, q90, alpha=0.25)
        plt.plot(dates, q50, label='median P')
        plt.title('Participation (P) forecast fan chart')
        plt.xlabel('date')
        plt.ylabel('Participation rate')
        plt.ylim(0,1)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # histogram of time-outside across sims
        plt.figure(figsize=(6,3))
        P = P_paths
        outside = ((P < params()['Plow']) | (P > params()['Phigh'])).sum(axis=1)
        plt.hist(outside, bins=40)
        plt.title('Distribution of days outside D0 across simulations')
        plt.xlabel('days outside D0 over horizon')
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()

        # print quick numbers
        print('\nQuick summary:')
        print(f"Median path start P: {float(P[:,0].mean()):.3f}")
        print(f"Median P after horizon: {float(np.median(P[:,-1])):.3f}")

    #dates = np.arange(parameters['horizon_days'] + 1)
    #result_diagnostics = diagnostics(dates, P_paths, result_risk)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
