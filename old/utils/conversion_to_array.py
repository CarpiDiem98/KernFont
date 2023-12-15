import numpy as np
import pandas as pd
import ast


def array_conversion(path):
    """_summary_

    Args:
        path (str): path of the file to convert

    Returns:
        pandas.DataFrame: _description_
    """
    df = pd.read_csv(path)
    try:
        df['X1'] = [np.squeeze(np.array(ast.literal_eval(row.replace(
            'array(', '').replace(')', ''))).transpose(0, 2, 1)) for row in df['X1']]
        df['X2'] = [np.squeeze(np.array(ast.literal_eval(row.replace(
            'array(', '').replace(')', ''))).transpose(0, 2, 1)) for row in df['X2']]
    except Exception as e:
        try:
            df['X1'] = [np.array(ast.literal_eval(row.replace('\n', '').replace(
                ' ', ',').replace(',,', ','))) for row in df['X1']]
            df['X2'] = [np.array(ast.literal_eval(row.replace('\n', '').replace(
                ' ', ',').replace(',,', ','))) for row in df['X2']]
        except Exception as e:
            print(e)
            pass
    return df
