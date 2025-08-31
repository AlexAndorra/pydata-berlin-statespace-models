# A Beginner's Guide to State Space Modeling

> A hands-on [PyData tutorial](https://cfp.pydata.org/berlin2025/talk/GRZ3RG/), by [Alexandre Andorra](https://www.linkedin.com/in/alex-andorra/) and [Jesse Grabowski](https://www.linkedin.com/in/jessegrabowski/)

We believe this tutorial will empower participants with practical knowledge of state space modeling in PyMC, enabling them to effectively analyze complex time series data using Bayesian approaches.

## Setup

This tutorial can be set up using [Anaconda](https://www.anaconda.com/products/individual#download-section) or [Miniforge](https://github.com/conda-forge/miniforge#download), which is a lightweight version of Anaconda that is easier to work with.

### Getting the Course Materials

The next step is to clone or download the course materials. If you are familiar with Git, run:

    git clone https://github.com/AlexAndorra/pydata-berlin-statespace-models.git

otherwise you can [download a zip file](https://github.com/AlexAndorra/pydata-berlin-statespace-models/archive/refs/heads/main.zip) of its contents, and unzip it on your computer.

### Setting up the Environment

The repository contains an `environment.yml` file with all required packages. Run:

    mamba env create -f environment.yml

from the main course directory (use `conda` instead of `mamba` if you installed Anaconda). Then activate the environment:

    mamba activate pydata-ssm
    # or
    conda activate pydata-ssm

Then, you can start **JupyterLab** to access the materials:

    jupyter lab

For those who like to work in VS Code, you can also run Jupyter notebooks from within VS Code. To do this, you will need to install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter). Once this is installed, you can open the notebooks in the `notebooks` subdirectory and run them interactively.

---

## Abstract

**State Space Models** (SSMs) are powerful tools for time series analysis, widely used in finance, economics, ecology, and engineering. They allow researchers to encode structural behavior into time series models, including *trends*, *seasonality*, *autoregression*, and *irregular fluctuations*, to name just a few. Many workhorse time series models, including ARIMA, VAR, and ETS, are special cases of the general statespace framework.  

In this practical, hands-on tutorial, attendees will **learn how to leverage PyMC's new state-space modeling** capabilities ([`pymc_extras.statespace`](https://github.com/pymc-devs/pymc-extras/tree/main/pymc_extras)) to build, fit, and interpret Bayesian state space models.

Starting from fundamental concepts, we'll **explore several real-world use cases**, demonstrating how SSMs help tackle common time series challenges, such as handling missing observations, integrating external regressors, and generating forecasts.

## Description

State Space Models offer **a structured yet flexible framework for time series analysis**. They elegantly handle latent processes like trends, seasonality, and noisy observations, making them particularly valuable in real-world applications.

We'll start with a brief overview of the theory behind SSMs, followed by practical examples where participants will:

- **Understand the components of SSMs**, including observation and state equations.
- **Learn how to specify and fit SSMs** using PyMC's state space module.
- Implement a **modeling workflow using a survey data example**, showing how to use SSMs to model the data and generate predictions.
- **Explore advanced topics** such as incorporating external regressors, generating forecasts or building custom models.

### Target Audience

This tutorial is aimed at data scientists, statisticians, and data analysts with a basic understanding of statistics and Python, who are interested in expanding their toolkit with Bayesian time series methods. Prior experience with [PyMC](https://www.pymc.io/welcome.html) is not required but will be beneficial.

### Takeaways

By the end of this tutorial, attendees will:

- Understand the **theoretical foundations** of State Space Models.
- Be able to **implement common SSMs** (local level, trend, and seasonal models) in PyMC.
- **Evaluate and interpret** Bayesian state space models using PyMC.
- **Appreciate practical scenarios** where SSMs outperform traditional time series approaches. 

### Background Knowledge Required

Basic understanding of probability and statistics, and familiarity with Python. Prior experience with PyMC is not required but will be beneficial.

## Outline

**0 - 10 min: Introduction to State Space Models**

- What are SSMs, and why use them?

**10 - 25 min: State Space Model Fundamentals**

- Observation and state equations.
- Latent states, Kalman filters, and smoothing in Bayesian frameworks.

**25 - 55 min: Implementing SSMs with PyMC (Hands-On)**

- Setting up a local-level model in PyMC.
- Extending models to incorporate trends and seasonality.
- Posterior inference: interpreting results and uncertainty.

**55 - 75 min: Advanced State Space Modeling (Hands-On)**

- Dealing with missing data and irregular intervals.
- Adding external covariates (regression components).
- Model diagnostics and posterior predictive checks.

**75 - 85 min: Real-world Application Case Study**

- Demonstrating an end-to-end modeling example with real data.
- Discussing best practices for practical time series modeling.

**85 - 90 min: Wrap-up and Interactive Q&A**

- Open floor for questions and further resources.

---

## Additional Resources

- [Podcast episode on PyMC's state space module](https://learnbayesstats.com/episode/124-state-space-models-structural-time-series-jesse-grabowski)
- [Introduction to PyMC state space module](https://www.youtube.com/watch?v=G9VWXZdbtKQ)
- [PyMC State Space Module GitHub Repository](https://github.com/pymc-devs/pymc-extras/tree/main/pymc_extras/statespace)
