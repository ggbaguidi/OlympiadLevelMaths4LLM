"""
Prepare reference CSV for evaluation.
"""
from typing import Iterable, Optional, Union
import pandas as pd

def prepare_reference_csv(
    input_path: str,
    output_path: str = "reference.csv",
    problem_ids: Optional[Union[str, Iterable[str]]] = None,
):
    """
    Load reference data, optionally filter by problem id(s),
    store ground truth answers, and save a CSV without answers.

    Parameters
    ----------
    input_path : str
        Path to the original reference.csv
    output_path : str
        Path where the filtered reference.csv (without answers) is saved
    problem_ids : str | Iterable[str] | None
        - None: use all problems
        - str: single problem id
        - Iterable[str]: multiple problem ids

    Returns
    -------
    ground_truth : dict
        Mapping {id: answer} for the selected problems
    df_filtered : pd.DataFrame
        Filtered dataframe (with answers removed)
    """

    df = pd.read_csv(input_path)

    # Normalize problem_ids
    if problem_ids is not None:
        if isinstance(problem_ids, str):
            problem_ids = {problem_ids}
        else:
            problem_ids = set(problem_ids)

        df = df[df["id"].isin(problem_ids)]

    # Store ground truth answers (if present)
    ground_truth = (
        dict(zip(df["id"], df["answer"]))
        if "answer" in df.columns
        else {}
    )

    # Remove answers and save
    df_filtered = df.drop("answer", axis=1, errors="ignore")
    df_filtered.to_csv(output_path, index=False)

    return ground_truth, df_filtered