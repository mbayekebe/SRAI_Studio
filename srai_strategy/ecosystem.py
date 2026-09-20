"""AI ecosystem and international-cooperation helpers."""
from __future__ import annotations

def ecosystem_map(government,academia,private_sector,civil_society,development_partners):
    return {
        "government":list(government),
        "academia":list(academia),
        "private_sector":list(private_sector),
        "civil_society":list(civil_society),
        "development_partners":list(development_partners),
    }

def cooperation_priority(knowledge_transfer,funding,regional_alignment,local_ownership):
    return float(.30*knowledge_transfer+.20*funding+.20*regional_alignment+.30*local_ownership)

def partnership_principles():
    return [
        "local_ownership",
        "capacity_transfer",
        "interoperability",
        "sustainability",
        "responsible_data_use",
        "mutual_accountability",
    ]
