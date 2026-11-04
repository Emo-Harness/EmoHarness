"""The paper's reproduction pipeline: run every stage, then compute every reported number.

    python -m experiments.paper.pipeline all        # data -> mme -> mhe -> intentqa -> report
    python -m experiments.paper.report              # tables, figure data and numbers from results/

``pipeline`` drives the runs, ``mme``, ``construction`` and ``intentqa`` compute
the results of each part of the paper, ``report`` writes them all under
``results/paper/``, and ``cases`` prints the per-item records behind a case study.
"""
