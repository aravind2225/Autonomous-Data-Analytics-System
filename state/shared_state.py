from typing import TypedDict, List, Dict, Any
import pandas as pd


class SharedState(TypedDict, total=False):
    file: any
    uploaded_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    metadata: Dict[str, Any]
    cleaning_report: Dict[str, Any]
    eda_results: Dict[str, Any]
    statistical_results: Dict[str, Any]
    visualizations: Dict[str, Any]
    insights: List[str]
    chat_history: List[Dict[str, str]]
    logs: List[str]
    errors: List[str]
    workflow_status: str
    visualization_approved: bool
    insights_approved: bool