"""Quick state dump — list registered ontology specs and their structure."""

from schema_org_event import SCHEMA_ORG_EVENT
from sem_event import SEM_EVENT
from bfo_cco_event import BFO_CCO_EVENT


def main():
    specs = [SCHEMA_ORG_EVENT, SEM_EVENT, BFO_CCO_EVENT]

    for spec in specs:
        print(f"\n{'─' * 60}")
        print(f"  {spec.summary()}")
        print(f"{'─' * 60}")

        print("  Node types:")
        for name, ndef in spec.node_types.items():
            system = " [system]" if name in spec.system_managed_nodes else ""
            print(f"    {name:25s}  ← {ndef.ontology_mapping}{system}")

        print("  Relationship types:")
        for name, rdef in spec.relationship_types.items():
            system = " [system]" if name in spec.system_managed_rels else ""
            print(f"    {name:25s}  ({rdef.from_label} → {rdef.to_label}){system}")

        print(f"  Completeness rules: {len(spec.completeness_rules)}")
        for rule in spec.completeness_rules:
            print(f"    [{rule.priority:8s}] {rule.description}")


if __name__ == "__main__":
    main()
