"""Cleaning report: read runs/<run_id>/ → a single-file HTML (see html.build_report)."""

from .html import build_report, render_report
from .rundata import RunData, load_run, load_trace, run_to_dict

__all__ = ["RunData", "build_report", "load_run", "load_trace", "render_report", "run_to_dict"]
