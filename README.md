# AI-Driven Portfolio Optimizer

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Finance](https://img.shields.io/badge/Finance-Modern_Portfolio_Theory-green.svg)
![AI](https://img.shields.io/badge/LLM-GPT_4o_mini-orange.svg)
![Optimization](https://img.shields.io/badge/Optimization-SLSQP-red.svg)

An advanced investment portfolio optimization engine that merges **Modern Portfolio Theory (MPT)** with **AI-driven Sentiment Analysis**. The system calculates optimal asset weights by balancing historical risk-return profiles with real-time market sentiment extracted via Large Language Models (LLMs).

---

## Financial Methodology

* **Logarithmic Returns:** Instead of simple percentage changes, we utilize $ln(P_t / P_{t-1})$ to ensure time-additivity and to better model the stochastic nature of price movements.
* **Mean-Variance Optimization:** The core engine solves for the **Tangency Portfolio** by maximizing the **Sharpe Ratio**: 
    $$\text{Maximize } S_p = \frac{R_p - R_f}{\sigma_p}$$
    where $R_f$ is the Risk-Free Rate (configured at 4.5% to reflect current treasury yields).
* **Annualization Logic:** Calculations automatically adjust for a standard 252-day trading year to derive annual expected returns and the covariance matrix.
* **Optimization Constraints:** Uses the **SLSQP (Sequential Least Squares Programming)** algorithm with:
    * `Constraint 1`: Sum of weights equals 100% (Full investment).
    * `Constraint 2`: Weights between 0 and 1 (No short-selling / Long-only constraint).

## Tools Used

* **Asynchronous Concurrency (`asyncio`):** To solve the latency bottleneck of LLM processing, news sentiment analysis is performed concurrently using `asyncio.gather`, allowing for near-instant processing of multiple tickers.
* **Structured LLM Outputs:** Utilizes OpenAI's `response_format={ "type": "json_object" }` to ensure deterministic, machine-readable sentiment scores, eliminating the unpredictability of natural language parsing.
* **Data Integrity & Sanitization:**
    * Automated handling of missing price data using forward-fills and threshold-based dropping.
    * Robust integration with `yfinance` API with error handling for ticker-specific delistings or network timeouts.
* **Clean Architecture:** Implements immutable configurations using `dataclasses` and enforces strict **Type Hinting** across the entire codebase for maintainability and IDE support.

## Prerequisites
* Python 3.10+
* OpenAI API Key (for sentiment analysis)
