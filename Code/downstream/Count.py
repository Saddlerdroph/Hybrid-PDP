#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, List

import numpy as np


def downstream_query(retained_values: List[Any], tau_for_noise: float,beta:float) -> float:
    tau_for_noise = float(tau_for_noise)
    if tau_for_noise <= 0:
        raise ValueError("tau_for_noise must be > 0")

    true_count = float(len(retained_values))
    noise = float(np.random.laplace(loc=0.0, scale=(1.0 / tau_for_noise)))
    return true_count + noise
