"""Sustainable AI helpers."""
from __future__ import annotations

def sustainability_score(energy_efficiency,hardware_lifecycle,financial_sustainability,local_capacity,social_value):
    values=[energy_efficiency,hardware_lifecycle,financial_sustainability,local_capacity,social_value]
    return sum(values)/len(values)

def carbon_estimate(compute_hours,power_kw,carbon_intensity):
    return float(compute_hours*power_kw*carbon_intensity)

def sustainability_actions():
    return [
        "right_size_models",
        "reuse_shared_services",
        "measure_energy",
        "extend_hardware_lifecycle",
        "build_local_capacity",
        "fund_long_term_operations",
    ]
