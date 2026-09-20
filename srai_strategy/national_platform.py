"""National AI platform helpers."""
from __future__ import annotations

def national_ai_layers():
    return [
        "digital_identity_and_trust",
        "connectivity_and_cloud",
        "national_data_exchange",
        "shared_ai_platform",
        "sector_data_products",
        "public_service_applications",
        "governance_and_assurance",
    ]

def sovereignty_pattern(data_sensitivity,local_capacity,regional_partnership):
    if data_sensitivity=="high" and local_capacity=="high":
        return "sovereign_national"
    if local_capacity=="low" and regional_partnership:
        return "regional_shared_platform"
    return "hybrid_sovereign"

def public_value_score(inclusion,service_quality,efficiency,trust,resilience):
    values=[inclusion,service_quality,efficiency,trust,resilience]
    return sum(values)/len(values)

def platform_readiness(identity,data_exchange,compute,talent,governance):
    checks={"identity":identity,"data_exchange":data_exchange,"compute":compute,"talent":talent,"governance":governance}
    return {"checks":checks,"ready":all(checks.values()),"score":sum(bool(v) for v in checks.values())/len(checks)}
