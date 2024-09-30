from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .metrics import round_limit_metric
from .returns import to_prices, to_returns
from .shuffle import shuffle_in_chunks
from .stats import to_drawdown_series

ReturnsDataFrame = pd.DataFrame
Returns = pd.Series
Signal = pd.Series


@dataclass
class BacktestResult:
    returns: Returns
    transaction_costs: pd.Series


def backtest_signal(
    signal: Signal,
    underlying: Returns,
    transaction_cost: float,
    lag: int,
) -> BacktestResult:
    """
    Create returns from a signal and a target.
    """
    assert isinstance(signal, pd.Series), "Signal must be a Series"
    signal = signal.ffill()
    underlying = underlying[signal.index[0] :]
    delta_pos = signal.diff(1).abs().fillna(0.0)
    costs = transaction_cost * delta_pos
    returns = (underlying * signal.shift(1 + lag)) - costs
    if returns.var() == 0.0:
        return BacktestResult(
            pd.Series(0.0, index=returns.index),
            transaction_costs=pd.Series(0.0, index=returns.index),
        )
    return BacktestResult(returns=returns, transaction_costs=costs)


@dataclass
class PortfolioBacktestResult:
    portfolio_returns: Returns
    component_returns: ReturnsDataFrame
    fees: pd.Series
    portfolio_returns_after_fees: Returns
    lag: int

    def split(self, start_date, end_date) -> PortfolioBacktestResult:
        return PortfolioBacktestResult(
            portfolio_returns=self.portfolio_returns[start_date:end_date],
            component_returns=self.component_returns[start_date:end_date],
            fees=self.fees[start_date:end_date],
            portfolio_returns_after_fees=self.portfolio_returns_after_fees[
                start_date:end_date
            ],
            lag=self.lag,
        )


def backtest_portfolio(
    weights: pd.DataFrame,
    underlying: ReturnsDataFrame,
    transaction_cost: float,
    management_fee: float | None,
    high_watermark_performance_fee: float | None,
    lag: int,
    fee_accounting_period: str = "ME",
) -> PortfolioBacktestResult:
    """
    Create returns from a signal and a target.
    """
    assert weights.columns.equals(underlying.columns), "Columns must match"
    underlying = underlying.loc[weights.index]
    weights = weights.ffill().reindex(underlying.index).ffill().copy()
    weights.columns = underlying.columns
    delta_pos = weights.diff(1).abs().fillna(0.0)
    costs = transaction_cost * delta_pos
    returns = (underlying * weights.shift(1 + lag)) - costs
    portfolio_returns = returns.sum(axis="columns")

    performance_fees = (
        _calculate_performance_fees(
            portfolio_returns,
            high_watermark_performance_fee,
            fee_accounting_period,
        )
        if high_watermark_performance_fee is not None
        else pd.Series(0.0, index=returns.index)
    )
    management_fees = (
        _calculate_mgmt_fees(
            portfolio_returns,
            management_fee,
            fee_accounting_period,
        )
        if management_fee is not None
        else pd.Series(0.0, index=returns.index)
    )
    fees = performance_fees + management_fees

    return PortfolioBacktestResult(
        portfolio_returns=portfolio_returns,
        component_returns=returns,
        fees=fees,
        portfolio_returns_after_fees=portfolio_returns - fees,
        lag=lag,
    )


def _calculate_performance_fees(
    portfolio_returns: pd.Series,
    high_watermark_performance_fee: float,
    accounting_period: str,
) -> pd.Series:
    drawdown = to_drawdown_series(portfolio_returns.resample(accounting_period).last())
    drawdown[drawdown == 0.0] = 1.0
    drawdown[drawdown < 0.0] = 0.0
    fee_multiplier = drawdown * high_watermark_performance_fee
    return (
        (
            to_returns(
                to_prices(portfolio_returns).resample(accounting_period).last(),
                clip=None,
            ).clip(lower=0.0)
            * fee_multiplier
        )
        .reindex(portfolio_returns.index)
        .fillna(0.0)
    )


def _calculate_mgmt_fees(
    portfolio_returns: pd.Series,
    management_fee: float,
    accounting_period: str,
) -> pd.Series:
    resampled_returns = portfolio_returns.resample(accounting_period).last()
    return (
        pd.Series(management_fee, index=resampled_returns.index)
        .reindex(portfolio_returns.index)
        .fillna(0.0)
    )


def calculate_shuffled_signal_metric(
    signal: Signal,
    benchmark: Returns,
    chunk_size: int | float,
    runs: int,
    transaction_costs: float,
    func: Callable,  # either finml_utils.stats.sharpe or finml_utils.stats.sortino
    annualization_period: int,
    lag: int,
) -> list[float]:
    signal = signal.to_frame()
    return [
        round_limit_metric(
            func(
                backtest_signal(
                    shuffle_in_chunks(signal, chunk_size=chunk_size).squeeze(),
                    benchmark,
                    transaction_cost=transaction_costs,
                    lag=lag,
                ).returns,
                annualization_period=annualization_period,
            )
        )
        for _ in range(runs)
    ]
