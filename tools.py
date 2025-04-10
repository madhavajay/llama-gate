import logging
from pathlib import Path
from langchain_core.tools import tool
from helper import FileInfo, list_files_with_metadata
import pandas as pd

logger = logging.getLogger(__name__)

workspace_dir = Path("/Users/madhavajay/dev/llm-glove/chat/sandbox")

def sandbox_path(file_path):
    abs_path = Path(file_path).resolve()

    if not abs_path.is_relative_to(workspace_dir):
        logger.debug(
            f"Attempted access to {abs_path} outside sandbox {workspace_dir}"
        )
        raise ValueError(
            f"The provided path: {abs_path} is outside the allowed sandbox directory: {workspace_dir}."
        )

    return abs_path

@tool
def list_all_files_in_datasite() -> FileInfo:
    """
    Recursively lists all files in a directory and gathers metadata for each file.

    Returns:
        dict: A dictionary where each key is the file path and the value is another dictionary containing
              file metadata (hash, extension, size in MB, MIME type, and Magika group).
    """
    print("> running list_all_files_in_datasite")
    results = list_files_with_metadata(workspace_dir)
    print("got results", results)
    return results

@tool
def list_all_files_in_private() -> FileInfo:
    """
    Recursively lists all files in a directory and gathers metadata for each file.

    Returns:
        dict: A dictionary where each key is the file path and the value is another dictionary containing
              file metadata (hash, extension, size in MB, MIME type, and Magika group).
    """
    results = list_files_with_metadata(workspace_dir)
    return results

@tool
def get_csv_columns_and_types(csv_path) -> str:
    """
    Reads a CSV file using pandas and returns a dictionary containing
    the column names and their corresponding data types.

    Parameters:
    csv_path (str): The file path to the CSV file.

    Returns:
    dict: A dictionary where keys are column names and values are data types.
          Example: {'column1': dtype('int64'), 'column2': dtype('object')}
    """
    try:
        csv_path = sandbox_path(csv_path)
        logger.info(f"Getting CSV columns and types for {csv_path}")
        df = pd.read_csv(csv_path)
        column_info = {col: df[col].dtype for col in df.columns}
        return str(column_info)
    except Exception as e:
        logger.error(f"Error reading CSV columns: {e}")
        return {"status": "error", "message": str(e)}

@tool
def get_csv_head_as_dict(csv_path) -> str:
    """
    Reads a CSV file using pandas and returns the first `n` rows as a list of
    dictionaries, where each dictionary represents a row with column names as keys.

    Parameters:
    csv_path (str): The file path to the CSV file.

    Returns:
    list: A list of dictionaries, each representing a row from the CSV.
          Example: [{'column1': value1, 'column2': value2}, ...]
    """
    try:
        csv_path = sandbox_path(csv_path)
        logger.info(f"Getting CSV head as dict for {csv_path}")
        df = pd.read_csv(csv_path)
        head_data = df.head(5).to_dict(orient="records")
        return str(head_data)
    except Exception as e:
        logger.error(f"Error reading CSV head: {e}")
        return {"status": "error", "message": str(e)}

# Tool mapping for the LLM
tool_mapping = {
    "list_all_files_in_datasite": list_all_files_in_datasite,
    "get_csv_columns_and_types": get_csv_columns_and_types,
    "get_csv_head_as_dict": get_csv_head_as_dict,
    "list_all_files_in_private": list_all_files_in_private,
}

def get_tools():
    return list(tool_mapping.values()) 