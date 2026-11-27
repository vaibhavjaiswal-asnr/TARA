"""
Factory module for auto-detecting and loading nuclear data formats.
"""
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional

from common.validation import require_existing_file
from input_output.base import BaseFormatFile
from input_output.leapr.core import LeaprFile
from input_output.ncrystal.core import NCrystalFile


def detect_format(filepath: Union[str, Path], sniff_limit: int = 50) -> str:
    """Detects the file format by sniffing the first few lines for known signatures.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path identifying the input or temporary resource.
    sniff_limit : int, optional
        Maximum number of file lines inspected for format detection.

    Returns
    -------
    str
        One of "ncmat", "leapr", and "csv".

    Raises
    ------
    FileNotFoundError
        If the source path is missing or is not a file.
    """
    path = Path(filepath)
    require_existing_file(path, message=f"The file {path} does not exist.")

    # Open the external file with an explicit encoding and deterministic resource cleanup.
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if '@' in line:
                return 'ncmat'
            if i == 0 and 'leapr' in line.lower():
                return 'leapr'
            if i >= sniff_limit:
                break
                
    return 'csv'


def load_file_object(filepath: Union[str, Path]) -> Tuple[Optional[BaseFormatFile], pd.Series]:
    """
    Parses the file, returning both the format object (if applicable) and the PDOS.

    Parameters
    ----------
    filepath : str or pathlib.Path
        The path to the input file.

    Returns
    -------
    Tuple[Optional[BaseFormatFile], pd.Series]
        The parsed object template (None if CSV) and the raw density of states.
    """
    fmt = detect_format(filepath)
    path_str = str(filepath)

    if fmt == 'ncmat':
        nc_obj = NCrystalFile.from_file(path_str)
        return nc_obj, nc_obj.pdos

    elif fmt == 'leapr':
        leapr_obj = LeaprFile.from_file(path_str)
        return leapr_obj, leapr_obj.pdos

    elif fmt == 'csv':
        df = pd.read_csv(path_str, header=None, sep=None, engine='python')
        if df.shape[1] < 2:
            raise ValueError("CSV format requires at least two columns (Energy and PDOS).")
        pdos = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].values)
        return None, pdos
    
    raise ValueError(f"Unknown format routing error for {filepath}.")
