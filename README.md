# trading-strategy-backtest
Infrastructure to backtest trading strategies with zipline-reloaded.

## Installation

```
cp extension.py <ZIPLINE_ROOT>
ln -s sharadar_ingest.py <ZIPLINE_ROOT>
export NASDAQ_API_KEY=<API KEY>
```

## Ingest data

```
zipline ingest -b sharadar
```

## Environment variables

- `SHARADAR_START_DATE`
- `SHARADAR_END_DATE`
- `SHARADAR_USE_CACHED_TABLES`

To setup variables for conda env you can use:
```
conda env config vars set VAR=VALUE
```