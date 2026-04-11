import numpy as np
import yaml
import logging
import os
import time
try:
    from xbbg import blp
except ImportError:
    blp = None
import pandas as pd


def setup_logging(script_name: str, log_dir: str = None) -> None:
    """
    Setup logging configuration for a script.

    Parameters:
    -----------
    script_name : str
        Name of the script (used for log filename)
    log_dir : str, optional
        Directory to save log files. Defaults to ../log relative to pysrc
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log')

    mkdir(log_dir)

    log_file = os.path.join(log_dir, f'{script_name}.log')

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Reset the log file
    if os.path.exists(log_file):
        open(log_file, 'w').close()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized for {script_name}")


def print_completed_time(start_time: float, task_name: str = "Task") -> None:
    """
    Log the time taken to complete a task.

    Parameters:
    -----------
    start_time : float
        Start time from time.time()
    task_name : str
        Name of the task that was completed
    """
    elapsed = time.time() - start_time

    if elapsed < 1:
        logging.info(f"{task_name} completed in {elapsed:.3f} seconds")
    elif elapsed < 3600:
        minutes = elapsed / 60
        logging.info(f"{task_name} completed in {minutes:.2f} minutes")
    else:
        hours = elapsed / 3600
        logging.info(f"{task_name} completed in {hours:.2f} hours")


def get_params(block=None):
    """
    Load parameters from PARAMS.yaml file.

    Parameters:
    -----------
    block : str, optional
        Specific block/section to load from PARAMS.yaml. If None, loads entire file.

    Returns:
    --------
    dict
        Dictionary containing parameters
    """
    # Try multiple locations for PARAMS.yaml
    possible_paths = [
        os.path.join(os.getcwd(), 'PARAMS.yaml'),
        os.path.join(os.getcwd(), 'params.yaml'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'PARAMS.yaml'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'params.yaml')
    ]

    params_file = None
    for path in possible_paths:
        if os.path.exists(path):
            params_file = path
            break

    if params_file is None:
        logging.error('PARAMS.yaml does not exist in any expected location')
        return None

    logging.debug("get_params: params_file: {}".format(params_file))

    with open(params_file, 'r') as f:
        PARAMS = yaml.safe_load(f)

    if block is not None:
        if block in PARAMS:
            return PARAMS[block]
        else:
            logging.error(f'Block "{block}" not found in PARAMS.yaml')
            return None

    return PARAMS

def format_df_print(df, decimals=1, percent_cols=[], exclude_zeros=False, excluded_cols=[]):
    formatted_df = df.copy()
    is_not_excluded = df.columns[~df.columns.to_series().isin(excluded_cols)]
    if exclude_zeros:
        formatted_df.loc[:, is_not_excluded] = formatted_df.loc[~np.isclose(formatted_df, 0).all(axis=1), is_not_excluded]
    if len(percent_cols) > 0:
        formatted_df.loc[:, percent_cols] *= 100
    formatted_df = (formatted_df).round(decimals)
    if len(percent_cols) > 0 and decimals == 0:
        formatted_df.loc[:, percent_cols] = formatted_df.loc[:, percent_cols].astype('int64')
    if len(percent_cols) > 0:
        mapper = {}
        for percent_col in percent_cols:
            mapper[percent_col] = percent_col + ' (%)'
        formatted_df = formatted_df.rename(columns=mapper)
    return formatted_df

def get_bbg_data(bloomberg_code, flds, start_date=None, end_date=None, drop_flds=False, max_attempts=1):
    logging.debug("flds: " + str(flds))
    is_mult_tickers = type(bloomberg_code) in [list, pd.Series, np.array]
    
    if start_date is None and end_date is None:
        try:
            h1 =  blp.bdp(tickers=bloomberg_code, flds=flds)
        except Exception as e:
            logging.error("Error getting Bloomberg details for ", bloomberg_code, ": " + str(e))
            
            return None
        return h1
    
    try:
        h1 =  blp.bdh(tickers=bloomberg_code, flds=flds, start_date=start_date, end_date=end_date, QuoteType='P')
    except Exception as e:
        logging.error("Error getting Bloomberg details for ", bloomberg_code, ": " + str(e))
        return None
    
    for i in range(max_attempts):
        is_missing_tickers = is_mult_tickers and len(h1.columns) != len(bloomberg_code)
        if is_missing_tickers:
            miss_codes = pd.Series(bloomberg_code)[~pd.Series(bloomberg_code).isin(h1.columns.get_level_values(0))].to_list()
            h1 = pd.concat([h1] + [get_bbg_data(code, flds, start_date, end_date) for code in miss_codes], axis=1).sort_index()
        else:
            break

    is_missing_tickers = is_mult_tickers and len(h1.columns) != (len(bloomberg_code) if is_mult_tickers else 1)
    if is_missing_tickers:
        miss_codes = pd.Series(bloomberg_code)[~pd.Series(bloomberg_code).isin(h1.columns.get_level_values(0))].to_list()
        if len(miss_codes) == len(bloomberg_code):
            logging.warning('Could not find any of the Bloomberg codes')
        else:
            logging.warning('Could not find the following Bloomberg codes: ' + str(miss_codes))
    
    if drop_flds and h1.columns.nlevels > 1:
        h1 = h1.droplevel(1, axis=1)
        
    h1.index.name = 'Date'
    h1.index = pd.to_datetime(h1.index)
    
    logging.debug("Bloomberg details: " + str(h1))
    return h1

def get_start_end_dates(df):
    return pd.DataFrame({'Start' : df.apply(lambda srs: srs.first_valid_index()),
                         'End'   : df.apply(lambda srs: srs.last_valid_index())}).T

def get_first_valid_index_w_lookback(srs, lookback):
    first_valid_loc = srs.index.get_loc(srs.first_valid_index()) + lookback
    if first_valid_loc < len(srs.index):
        return srs.index[first_valid_loc]
    else: return np.nan

def get_start_dates_w_lookback(df, lookback):
    start_dates_w_lookback = df.apply(get_first_valid_index_w_lookback, lookback=lookback)
    return start_dates_w_lookback

def is_single_val(x):
    x_type = type(x)
    return (x_type is not list and x_type is not dict 
            and x_type is not np.array and x_type is not pd.Series and x_type is not pd.DataFrame)

def droplevel_except(df, except_col):
    except_cols = [except_col] if is_single_val(except_col) else except_col
    df = df.copy()
    ix_names = pd.Series(df.index.names)
    if ix_names is not None and len(ix_names) > 1:
        non_date_ix_names = ix_names[~ix_names.isin(except_cols)].values
        df = df.droplevel(non_date_ix_names)
    return df

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def pull_process_bbg_df(tickers, start_date, end_date, ticker_map):
    df = pd.DataFrame()
    for ticker in tickers:
        cur_df = get_bbg_data(ticker,'px_last', start_date, end_date)
        df = pd.concat([df, cur_df], axis=1)
    df = df.droplevel(1, axis=1).rename(columns=ticker_map)[list(ticker_map.values())].ffill()
    df = df.reindex(pd.bdate_range(df.index.min(), df.index.max()))
    df.index.name = 'Date'
    return df

def get_bbg_px_data(bloomberg_code_list, start_date, end_date, ffill=True):
    price_df = pd.DataFrame()
    for bloomberg_code in bloomberg_code_list:
        cur_bbg_df = get_bbg_data(bloomberg_code, 'PX_LAST', start_date, end_date, drop_flds=True)
        if not cur_bbg_df.empty:
            price_df = pd.concat([price_df, cur_bbg_df], axis=1)
    price_df.index.name = 'date'
    price_df.index = pd.to_datetime(price_df.index)
    if ffill:
        price_df = price_df.ffill()
    return price_df

def get_bbg_yield_data(bloomberg_code_list, start_date, end_date, ffill=True):
    price_df = pd.DataFrame()
    for bloomberg_code in bloomberg_code_list:
        cur_bbg_df = get_bbg_data(bloomberg_code, 'YLD_YTM_MID', start_date, end_date, drop_flds=True)
        price_df = pd.concat([price_df, cur_bbg_df], axis=1)
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()
    price_df.index.name = 'Date'
    if ffill:
        price_df = price_df.ffill()
    return price_df

def invert_dict(my_map):
    inv_map = {v: k for k, v in my_map.items()}
    return inv_map

def get_roll_code(roll_type='expiration', roll_days_before=3, adj_method='ratio'):
    
    if roll_type.lower() in ['expiration']:
        roll_type_code = 'R'
    elif roll_type.lower().replace(' ', '_') in ['first_notice']:
        roll_type_code = 'N'
    else:
        raise ValueError('Invalid roll type: ' + str(roll_type))
    
    if adj_method is None or adj_method.lower() in ['none']:
        adj_method_code = 'N'
    if adj_method.lower() in ['ratio']:
        adj_method_code = 'R'
    elif adj_method.lower() in ['difference', 'diff']:
        adj_method_code = 'D'
    else:
        raise ValueError('Invalid adjustment method: ' + str(adj_method))
    
    roll_code = f'{roll_type_code}:0{roll_days_before}_0_{adj_method_code}'
    return roll_code
    
def add_roll_code(fut_bbg_code_long, roll_type='expiration', roll_days_before=3, adj_method='ratio'):
    roll_code = get_roll_code(roll_type=roll_type, roll_days_before=roll_days_before, adj_method=adj_method)
    bbg_code_left, bbg_code_right = fut_bbg_code_long.rsplit(' ', 1)
    roll_bbg_code_long = bbg_code_left + ' ' + roll_code + ' ' + bbg_code_right
    return roll_bbg_code_long

def convert_int_to_tdelta(int_num, is_before):
    if is_before and int_num < 0:
        sign = '+'
    elif is_before and int_num >= 0:
        sign = '-'
    elif not is_before and int_num < 0:
        sign = '-'
    elif not is_before and int_num >= 0:
        sign = '+'
    return 't' + sign + str(abs(int_num))

def get_env_var(env_var_name, is_numeric=True):
    if is_numeric:
        if env_var_name in os.environ:
            env_var = os.environ[env_var_name]
            if int(env_var) == 1 or env_var == "True":
                env_var = True
            else:
                env_var = False
        else:
            env_var = False
    else:
        if env_var_name in os.environ:
            env_var = os.environ[env_var_name]
        else:
            env_var = None
    return env_var