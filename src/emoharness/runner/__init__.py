"""Orchestration layer: project paths, run context, row dispatch, and entry points.

Everything under ``examples/`` imports from here and nothing else. Heavy adapters
(vLLM, Gemini, faster-whisper, openSMILE) are imported lazily inside the factory
that needs them, so ``--fake`` runs never touch a GPU, a key, or the network.
"""
