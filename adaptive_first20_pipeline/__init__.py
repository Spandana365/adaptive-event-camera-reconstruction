"""Adaptive first-20 reconstruction pipeline for the fast-motion sequence."""


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)


__all__ = ["run_experiment"]
