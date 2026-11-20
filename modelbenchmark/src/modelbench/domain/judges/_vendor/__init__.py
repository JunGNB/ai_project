"""Vendored third-party judge assets shipped with the package.

Only code small enough to vendor lives here (currently the official Google
IFEval checker, ~96 KB of Python).  Large or network-only assets (NLTK punkt
data, upstream jsonl) stay under ``dataset/raw`` and are fetched by
``tool/dataset/accuracy/fetch.py``; ``./modelbench setup`` runs that fetch for
you so a fresh install can judge accuracy runs without extra manual steps.
"""
