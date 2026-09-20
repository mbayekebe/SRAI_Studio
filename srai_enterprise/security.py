"""Identity, authorization, and zero-trust helpers."""
from __future__ import annotations

def role_allows(role_permissions,role,permission):
    return permission in set(role_permissions.get(role,[]))

def attribute_policy(user,resource,action):
    if user.get("suspended"):
        return False
    if resource.get("classification")=="restricted":
        return user.get("clearance")=="high" and action in {"read","analyze"}
    if action=="delete":
        return "administrator" in user.get("roles",[])
    return True

def authorize(user,resource,action,role_permissions):
    role_decision=any(
        role_allows(role_permissions,role,f"{action}:{resource['type']}")
        for role in user.get("roles",[])
    )
    attribute_decision=attribute_policy(user,resource,action)
    return {
        "allowed":bool(role_decision and attribute_decision),
        "role_decision":bool(role_decision),
        "attribute_decision":bool(attribute_decision),
    }

def zero_trust_score(identity_verified,device_compliant,network_trusted,least_privilege,continuous_monitoring):
    values=[identity_verified,device_compliant,network_trusted,least_privilege,continuous_monitoring]
    return sum(bool(v) for v in values)/len(values)

def audit_event(user_id,action,resource_id,allowed,reason=None):
    return {
        "user_id":user_id,
        "action":action,
        "resource_id":resource_id,
        "allowed":bool(allowed),
        "reason":reason,
    }
