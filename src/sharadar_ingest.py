import os
import logging
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import requests
import pandas as pd
from click import progressbar
from pathlib import Path
from urllib.parse import urlencode
from zipline.utils.paths import data_root, cache_root
from zipline.pipeline.classifiers import Classifier, CustomClassifier
from zipline.utils.numpy_utils import int64_dtype
from zipline.utils.numpy_utils import object_dtype
from zipline.data import bundles
import zipline.utils.paths as pth
from zipline.lib.labelarray import LabelArray


np.random.seed(42)
pd.set_option('display.expand_frame_repr', False)

ONE_MEGABYTE = 1024 * 1024
NASDAQ_DATALINK_URL = 'https://data.nasdaq.com/api/v3/datatables/'
SHARADAR_BUNDLE_NAME = 'sharadar'
SHARADAR_BUNDLE_DIR = 'latest'

log = logging.getLogger(__name__)

def get_data_filepath(name, environ=None):
    """
    Returns a handle to data file.
    Creates containing directory, if needed.
    """
    dr = data_root(environ)

    if not os.path.exists(dr):
        os.makedirs(dr)

    return os.path.join(dr, name)

def get_cache_filepath(name, environ=None):
    """
    Returns a handle to cache file.
    Creates containing directory, if needed.
    """
    dr = cache_root(environ)

    if not os.path.exists(dr):
        os.makedirs(dr)

    return os.path.join(dr, name)

def get_output_dir():
    return os.path.join(get_data_filepath(SHARADAR_BUNDLE_NAME), SHARADAR_BUNDLE_DIR)

def get_cache_dir():
    cache_dir = os.path.join(get_cache_filepath(SHARADAR_BUNDLE_NAME), SHARADAR_BUNDLE_DIR)
    
    if not os.path.exists(cache_dir):
        logging.info(f"Folder '{cache_dir}' created.")
        os.makedirs(cache_dir)
    
    return cache_dir


def download_with_progress(url, chunk_size, **progress_kwargs):
    """
    Download streaming data from a URL, printing progress information to the
    terminal.

    Parameters
    ----------
    url : str
        A URL that can be understood by ``requests.get``.
    chunk_size : int
        Number of bytes to read at a time from requests.
    **progress_kwargs
        Forwarded to click.progressbar.

    Returns
    -------
    data : BytesIO
        A BytesIO containing the downloaded data.
    """
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total_size = int(resp.headers['content-length'])
    data = BytesIO()
    with progressbar(length=total_size, **progress_kwargs) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            data.write(chunk)
            pbar.update(len(chunk))

    data.seek(0)
    return data


def format_metadata_url(api_key, table_name):
    """ Build the query URL for Sharadar Prices metadata.
    """
    query_params = [('api_key', api_key), ('qopts.export', 'true')]

    return NASDAQ_DATALINK_URL + table_name + ".csv?" + urlencode(query_params)


def load_data_table(file, index_col=None, parse_dates=False):
    """ Load data table from zip file provided by Sharadar.
    """
    with ZipFile(file) as zip_file:
        file_names = zip_file.namelist()
        assert len(file_names) == 1, "Expected a single file from Sharadar."
        wiki_prices = file_names.pop()
        with zip_file.open(wiki_prices) as table_file:
            data_table = pd.read_csv(table_file, index_col=index_col,
                                     parse_dates=parse_dates, na_values=['NA'],
                                     converters={'ticker': str})

    return data_table


def fetch_entire_table(api_key, table_name, index_col=None, parse_dates=False, retries=5):
    zip_file = os.path.join(get_cache_dir(), os.path.split(table_name)[-1] + '.zip')
    
    if os.environ.get('SHARADAR_USE_CACHED_TABLES', None) and os.path.exists(zip_file):
        logging.info("Parsing data from file %s." % zip_file)
        return load_data_table(zip_file, index_col=index_col, parse_dates=parse_dates)

    logging.info("Start loading the entire %s dataset..." % table_name)
    for _ in range(retries):
        try:
            source_url = format_metadata_url(api_key, table_name)
            metadata = pd.read_csv(source_url)

            # Extract link from metadata and download zip file.
            table_url = metadata.loc[0, 'file.link']

            raw_file = download_with_progress(
                table_url,
                chunk_size=ONE_MEGABYTE,
                label="Downloading data from Sharadar table " + table_name
            )

            with open(zip_file, "wb") as f:
                logging.info("Saving loaded data to file %s." % zip_file)
                f.write(raw_file.getbuffer())

            logging.info("Parsing data from nasdaqdatalink table %s." % table_name)
            return load_data_table(raw_file, index_col=index_col, parse_dates=parse_dates)

        except Exception:
            logging.exception("Exception raised reading Sharadar data. Retrying.")

    else:
        raise ValueError("Failed to download data from '%s' after %d attempts." % (source_url, retries))
        
def fetch_data():
    """
    Fetch the Sharadar Equity Prices (SEP) and Sharadar Fund Prices (SFP). Entire dataset.
    """
    df_tic = fetch_entire_table(os.environ["NASDAQ_API_KEY"], "SHARADAR/TICKERS")
    df_sep = fetch_entire_table(os.environ["NASDAQ_API_KEY"], "SHARADAR/SEP", parse_dates=['date']) # Sharadar Equity Prices
    df_sfp = None #fetch_entire_table(os.environ["NASDAQ_API_KEY"], "SHARADAR/SFP", parse_dates=['date']) # Sharadar Fund Prices
    df_act = fetch_entire_table(os.environ["NASDAQ_API_KEY"], "SHARADAR/ACTIONS", parse_dates=['date'])

    return df_tic, df_sep, df_sfp, df_act


def adjust_prices(df):
    # 'close' prices are adjusted only for stock splits, but not for dividends.
    m = df['closeunadj'] / df['close']

    # Remove the split factor to get back the unadjusted data
    df['open'] *= m
    df['high'] *= m
    df['low'] *= m
    df['close'] = df['closeunadj']
    df['volume'] /= m
    df = df.drop(['closeunadj', 'closeadj', 'lastupdated'], axis=1)
    
    gaps = df.isin([np.nan, np.inf, -np.inf])
    #if gaps.sum().any():
    #    print("Nan/inf values found in prices for:")
    #    print(df[gaps.any(axis=1)].index.unique(level=0))
    
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    #df = df.fillna({'volume': 0})
    #df = df.ffill(limit=5).dropna(axis=1)
    return df
        

def prepare_prices_df(df, from_date, to_date):
    idx = pd.IndexSlice
    prices = df.set_index(['ticker', 'date']).sort_index().loc[idx[:, from_date : to_date], :]
    prices = adjust_prices(prices)
    names = prices.index.names
    prices = (prices
              .reset_index()
              .drop_duplicates()
              .set_index(names)
              .sort_index())
    
    # print('\nNo. of observations per asset')
    # print(prices.groupby('ticker').size().describe())
    # print(prices.info(show_counts=True))
    
    return prices


def gen_asset_metadata(symbols, prices, show_progress):
    if show_progress:
        log.info("Generating asset metadata.")

    #data = prices.index.to_frame().groupby(level=0).agg({"date": [np.min, np.max]})
    data = prices.index.to_frame().groupby(level=0).agg(start_date=('date', 'min'), end_date=('date', 'max'))
    #data["start_date"] = data.date.amin
    #data["end_date"] = data.date.amax
    #del data["date"]
    #data.columns = data.columns.get_level_values(0)
    data["auto_close_date"] = data["end_date"].values + pd.Timedelta(days=1)

    data = symbols.join(data, how='inner').reset_index().set_index('sid')
    data["exchange"] = "SHARADAR"
    data.rename(columns={'ticker': 'symbol', 'name': 'asset_name'}, inplace=True)
    return data

def parse_pricing_and_vol(data, sessions, symbol_map):
    for asset_id, symbol in symbol_map.items():
        asset_data = (
            data.xs(symbol, level=0).reindex(sessions.tz_localize(None)).fillna(0.0)
        )
        yield asset_id, asset_data

        
def parse_splits_df(splits, symbol_map, sid_map, show_progress):
    if show_progress:
        log.info("Parsing splits data.")
    
    # Filter entries
    splits = splits.loc[splits['ticker'].isin(symbol_map.values)].copy()

    # The date dtype is already datetime64[ns]
    splits['value'] = 1.0 / splits.value
    splits.rename(
        columns={
            'value': 'ratio',
            'date': 'effective_date',
        },
        inplace=True,
        copy=False,
    )
    splits['ratio'] = splits['ratio'].astype(float)
    splits['sid'] = splits['ticker'].apply(lambda x: sid_map.loc[x])
    splits.drop(['action', 'name', 'contraticker', 'contraname', 'ticker'], axis=1, inplace=True)
    
    gaps = splits.isin([np.nan, np.inf, -np.inf])
    if gaps.sum().any():
        log.info("Nan/inf values found in splits for:")
        log.info(symbol_map.loc[splits[gaps.any(axis=1)].sid.unique()])

    return splits


def parse_dividends_df(dividends, symbol_map, sid_map, show_progress):
    if show_progress:
        log.info("Parsing dividend data.")
    
    # Filter entries
    dividends = dividends.loc[dividends['ticker'].isin(symbol_map.values)].copy()

    # Modify for adjustment writer
    dividends = dividends.rename(columns={'value': 'amount'})
    dividends['sid'] = dividends['ticker'].apply(lambda x: sid_map.loc[x])
    dividends['ex_date'] = dividends['date']
    dividends.set_index('date', inplace=True)
    dividends["record_date"] = dividends["declared_date"] = dividends["pay_date"] = dividends['ex_date']
    dividends.drop(['action', 'name', 'contraticker', 'contraname', 'ticker'], axis=1, inplace=True)
    
    gaps = dividends.isin([np.nan, np.inf, -np.inf])
    if gaps.sum().any():
        log.info("Nan/inf values found in splits for:")
        log.info(symbol_map.loc[dividends[gaps.any(axis=1)].sid.unique()])
    
    return dividends
        

def sharadar_to_bundle(interval='1d'):
    def ingest(environ,
               asset_db_writer,
               minute_bar_writer,
               daily_bar_writer,
               adjustment_writer,
               calendar,
               start_session,
               end_session,
               cache,
               show_progress,
               output_dir
               ):
        
        log.info("Ingesting sharadar data from {} to {}".format(start_session, end_session))
                
        # Add code which loads raw data
        df_tic, df_sep, df_sfp, df_act = fetch_data()
        
        prices = prepare_prices_df(df_sep, start_session, end_session)
        tickers = prices.index.unique('ticker')
        symbols = (df_tic[(df_tic.table == "SEP") & df_tic.ticker.isin(tickers)]
                   .set_index('ticker')
                   .rename(columns={'permaticker': 'sid'}))

        #print(symbols.info(show_counts=True))
        
        # Write metadata
        metadata = gen_asset_metadata(symbols, prices, show_progress)
        exchanges = pd.DataFrame(
            data=[["SHARADAR", "SHARADAR", "US"]],
            columns=["exchange", "canonical_name", "country_code"],
        )
        metadata.to_hdf(Path(output_dir) / 'sharadar.h5', 'metadata', format='t')
        asset_db_writer.write(equities=metadata, exchanges=exchanges)

        symbol_map = metadata.symbol
        sid_map = symbol_map.reset_index().set_index('symbol')
        
        # Write prices
        sessions = calendar.sessions_in_range(start_session, end_session)
        daily_bar_writer.write(
            parse_pricing_and_vol(prices, sessions, symbol_map),
            assets=set(symbol_map.index),
            show_progress=show_progress
        )
        
        # Write splits and dividends
        splits_index = (
            (df_act.action == 'split')
            & (df_act.date >= start_session)
            & (df_act.date <= end_session)
            & (df_act.value != 1.0)
            & (df_act.value != 0.0)
        )
        splits_raw = df_act[splits_index]
        splits = parse_splits_df(splits_raw, symbol_map, sid_map, show_progress)

        dividends_raw = df_act[(df_act.action == 'dividend') & (df_act.date >= start_session) & (df_act.date <= end_session)]
        dividends = parse_dividends_df(dividends_raw, symbol_map, sid_map, show_progress)
        
        adjustment_writer.write(splits = splits, dividends = dividends)
    

    return ingest


class Sector(CustomClassifier):
    dtype = object_dtype
    window_length = 1
    inputs = []
    missing_value = 'NA'

    def __init__(self, timestamp=None):
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        all_ts = bundles.ingestions_for_bundle(SHARADAR_BUNDLE_NAME)
        ingest_ts = list(filter(lambda x: x <= timestamp, all_ts))
        dirname = bundles.to_bundle_ingest_dirname(ingest_ts[0])
        data_path = pth.data_path([SHARADAR_BUNDLE_NAME, dirname])
                 
        self.metadata = pd.read_hdf(Path(data_path)/ 'sharadar.h5', 'metadata')
        self.categories = ['Healthcare', 'Basic Materials', 'Financial Services', 'Consumer Cyclical', 'Technology',
            'Consumer Defensive', 'Industrials', 'Real Estate', 'Energy', 'Communication Services',
            'Utilities']
    
    def _allocate_output(self, windows, shape):
        return LabelArray(np.full(shape, self.missing_value), self.missing_value, categories=self.categories)
    
        
    def compute(self, today, assets, out, *arrays):
        data = self.metadata[['sector']].reindex(assets, fill_value='NA').T.values
        out[:] = LabelArray(data, self.missing_value, categories=self.categories)