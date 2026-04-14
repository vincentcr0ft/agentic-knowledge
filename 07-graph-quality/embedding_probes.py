"""
Phase 3 — Knowledge graph embedding probes via PyKEEN.

Requires: pip install pykeen

Uses KG embeddings (ComplEx) to:
  1. Predict missing links — surface potential gaps in the graph
  2. Score triple plausibility — flag low-confidence triples
  3. Cluster entities — detect misplaced or incorrectly linked nodes

This is most valuable for larger graphs (50+ triples), especially
in the multi-statement fusion scenario where multiple witness
accounts are combined.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quality_probe.core import DimensionResult, Violation


def _pykeen_available() -> bool:
    """Check if PyKEEN is installed."""
    try:
        import pykeen  # noqa: F401
        return True
    except Exception:
        return False


def _export_triples(driver) -> list[list[str]]:
    """Export Neo4j relationships as [subject, predicate, object] triples
    suitable for PyKEEN.

    Entities are encoded as 'Label:description' strings.
    """
    triples = []
    with driver.session() as session:
        result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN labels(a)[0] + ':' + "
            "  coalesce(a.description, a.name_or_description, a.value, "
            "    a.summary, toString(id(a))) AS subj, "
            "  type(r) AS pred, "
            "  labels(b)[0] + ':' + "
            "  coalesce(b.description, b.name_or_description, b.value, "
            "    b.summary, toString(id(b))) AS obj"
        )
        for rec in result:
            triples.append([rec["subj"], rec["pred"], rec["obj"]])
    return triples


def _train_model(triples: list[list[str]]):
    """Train a ComplEx model on the given triples via PyKEEN.

    Returns (model, triples_factory).
    """
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    import numpy as np

    arr = np.array(triples)
    tf = TriplesFactory.from_labeled_triples(arr)

    result = pipeline(
        training=tf,
        testing=tf,  # use training as test set (we score known triples)
        model="ComplEx",
        model_kwargs=dict(embedding_dim=50),
        training_kwargs=dict(
            num_epochs=100,
            batch_size=min(64, len(triples)),
        ),
        random_seed=42,
        use_tqdm=False,
    )
    return result.model, tf


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Link Prediction — discover missing relationships
# ═══════════════════════════════════════════════════════════════════════════════

def probe_link_prediction(driver, top_k: int = 10) -> DimensionResult:
    """Train a KG embedding model and predict missing links.

    Missing links suggest gaps: relationships that the graph structure
    implies should exist but don't.

    Returns predictions as info-level violations (they're suggestions,
    not errors).
    """
    if not _pykeen_available():
        return DimensionResult(
            dimension="link_prediction",
            score=-1.0,
            violations=[Violation(
                dimension="link_prediction",
                severity="info",
                message=(
                    "PyKEEN not installed — link prediction skipped. "
                    "Install with: pip install pykeen"
                ),
            )],
            details={"skipped": True, "reason": "pykeen not installed"},
        )

    import numpy as np

    triples = _export_triples(driver)

    if len(triples) < 10:
        return DimensionResult(
            dimension="link_prediction",
            score=1.0,
            violations=[Violation(
                dimension="link_prediction",
                severity="info",
                message=(
                    f"Graph has only {len(triples)} triples — too small for "
                    f"meaningful link prediction (need ≥ 10)"
                ),
            )],
            details={"triple_count": len(triples), "skipped": True},
        )

    try:
        import torch

        model, tf = _train_model(triples)
        model.eval()

        existing = set(tuple(t) for t in triples)
        predictions = []

        # For a sample of (head, relation) pairs, predict the best tails
        sample_size = min(20, len(triples))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(triples), size=sample_size, replace=False)

        entity_ids = tf.entity_to_id
        relation_ids = tf.relation_to_id
        id_to_entity = tf.entity_id_to_label

        for idx in sample_idx:
            h_label, r_label, _ = triples[idx]
            if h_label not in entity_ids or r_label not in relation_ids:
                continue
            h_id = entity_ids[h_label]
            r_id = relation_ids[r_label]

            with torch.no_grad():
                # Build (h, r) repeated for every possible tail
                n_ent = tf.num_entities
                hr = torch.tensor([[h_id, r_id]], dtype=torch.long).repeat(n_ent, 1)
                t_ids = torch.arange(n_ent, dtype=torch.long).unsqueeze(1)
                hrt = torch.cat([hr, t_ids], dim=1)
                scores = model.score_hrt(hrt).squeeze()
                top_k_ids = torch.topk(scores, min(5, n_ent)).indices.tolist()

            for t_id in top_k_ids:
                t_label = id_to_entity[t_id]
                candidate = (h_label, r_label, t_label)
                if candidate not in existing:
                    predictions.append(candidate)

        # Deduplicate and limit
        seen = set()
        unique_predictions = []
        for p in predictions:
            if p not in seen:
                seen.add(p)
                unique_predictions.append(p)
        predictions = unique_predictions[:top_k]

        violations = []
        for subj, rel, obj in predictions:
            violations.append(Violation(
                dimension="link_prediction",
                severity="info",
                message=f"Predicted missing link: ({subj})-[{rel}]->({obj})",
            ))

        missing_ratio = len(predictions) / max(len(triples), 1)
        score = max(0.0, 1.0 - missing_ratio)

        return DimensionResult(
            dimension="link_prediction",
            score=score,
            violations=violations,
            details={
                "triple_count": len(triples),
                "entity_count": tf.num_entities,
                "relation_count": tf.num_relations,
                "predicted_missing": len(predictions),
                "model": "ComplEx",
            },
        )

    except Exception as e:
        return DimensionResult(
            dimension="link_prediction",
            score=0.5,
            violations=[Violation(
                dimension="link_prediction",
                severity="info",
                message=f"Link prediction failed: {e}",
            )],
            details={"error": str(e)},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Triple Plausibility — score existing triples
# ═══════════════════════════════════════════════════════════════════════════════

def probe_triple_plausibility(driver, threshold: float = 0.3) -> DimensionResult:
    """Score the plausibility of each existing triple using KG embeddings.

    Low-scoring triples may be extraction errors or hallucinations.
    """
    if not _pykeen_available():
        return DimensionResult(
            dimension="triple_plausibility",
            score=-1.0,
            violations=[Violation(
                dimension="triple_plausibility",
                severity="info",
                message="PyKEEN not installed — plausibility scoring skipped",
            )],
            details={"skipped": True},
        )

    import numpy as np

    triples = _export_triples(driver)

    if len(triples) < 10:
        return DimensionResult(
            dimension="triple_plausibility",
            score=1.0,
            violations=[Violation(
                dimension="triple_plausibility",
                severity="info",
                message=f"Graph too small ({len(triples)} triples) for plausibility scoring",
            )],
            details={"triple_count": len(triples), "skipped": True},
        )

    try:
        import torch

        model, tf = _train_model(triples)
        model.eval()

        # Score each existing triple
        mapped = tf.mapped_triples
        with torch.no_grad():
            scores = model.score_hrt(mapped)

        # Normalise with sigmoid
        normalised = torch.sigmoid(scores).squeeze().numpy()

        violations = []
        low_scoring = []
        for i, (triple, score_val) in enumerate(zip(triples, normalised)):
            if score_val < threshold:
                low_scoring.append((triple, float(score_val)))
                violations.append(Violation(
                    dimension="triple_plausibility",
                    severity="warning",
                    message=(
                        f"Low-plausibility triple ({score_val:.3f}): "
                        f"({triple[0]})-[{triple[1]}]->({triple[2]})"
                    ),
                ))

        above_threshold = sum(1 for s in normalised if s >= threshold)
        plausibility_score = above_threshold / len(normalised) if len(normalised) > 0 else 1.0

        return DimensionResult(
            dimension="triple_plausibility",
            score=plausibility_score,
            violations=violations,
            details={
                "triple_count": len(triples),
                "low_plausibility_count": len(low_scoring),
                "mean_score": float(np.mean(normalised)),
                "min_score": float(np.min(normalised)),
                "threshold": threshold,
                "model": "ComplEx",
            },
        )

    except Exception as e:
        return DimensionResult(
            dimension="triple_plausibility",
            score=0.5,
            violations=[Violation(
                dimension="triple_plausibility",
                severity="info",
                message=f"Triple plausibility scoring failed: {e}",
            )],
            details={"error": str(e)},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Entity Clustering — detect misplaced entities
# ═══════════════════════════════════════════════════════════════════════════════

def probe_entity_clusters(driver) -> DimensionResult:
    """Use KG embeddings to cluster entities and detect outliers.

    Entities that cluster far from their expected group may be
    incorrectly linked or represent entity resolution failures
    (e.g., the same person extracted as two different nodes).
    """
    if not _pykeen_available():
        return DimensionResult(
            dimension="entity_clustering",
            score=-1.0,
            violations=[Violation(
                dimension="entity_clustering",
                severity="info",
                message="PyKEEN not installed — entity clustering skipped",
            )],
            details={"skipped": True},
        )

    import numpy as np

    triples = _export_triples(driver)

    if len(triples) < 15:
        return DimensionResult(
            dimension="entity_clustering",
            score=1.0,
            violations=[Violation(
                dimension="entity_clustering",
                severity="info",
                message=f"Graph too small ({len(triples)} triples) for clustering",
            )],
            details={"triple_count": len(triples), "skipped": True},
        )

    try:
        import torch

        model, tf = _train_model(triples)
        model.eval()

        # Get entity embeddings from PyKEEN
        entity_repr = model.entity_representations[0]
        with torch.no_grad():
            embeddings_t = entity_repr()
        embeddings = embeddings_t.detach().numpy()

        # Use real part only for ComplEx (embeddings are complex)
        if np.iscomplexobj(embeddings):
            embeddings = embeddings.real

        entities = [tf.entity_id_to_label[i] for i in range(tf.num_entities)]

        # Group entities by their Neo4j label (prefix before ':')
        label_groups: dict[str, list[int]] = {}
        for i, ent in enumerate(entities):
            label = ent.split(":")[0] if ":" in ent else "Unknown"
            label_groups.setdefault(label, []).append(i)

        violations = []

        # For each group, compute centroid and flag outliers (> 2 std from centroid)
        for label, indices in label_groups.items():
            if len(indices) < 2:
                continue

            group_embeddings = embeddings[indices]
            centroid = np.mean(group_embeddings, axis=0)
            distances = np.linalg.norm(group_embeddings - centroid, axis=1)

            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            if std_dist < 1e-8:
                continue

            for idx_in_group, global_idx in enumerate(indices):
                if distances[idx_in_group] > mean_dist + 2 * std_dist:
                    violations.append(Violation(
                        dimension="entity_clustering",
                        severity="warning",
                        message=(
                            f"Entity outlier: '{entities[global_idx]}' is distant "
                            f"from other {label} nodes (dist={distances[idx_in_group]:.3f}, "
                            f"group mean={mean_dist:.3f}) — possible extraction error"
                        ),
                        node_label=label,
                    ))

        # Also check for potential duplicates: entities of the same label
        # whose embeddings are very close
        for label, indices in label_groups.items():
            if len(indices) < 2:
                continue

            group_embeddings = embeddings[indices]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    dist = np.linalg.norm(group_embeddings[i] - group_embeddings[j])
                    if dist < 0.1:
                        violations.append(Violation(
                            dimension="entity_clustering",
                            severity="info",
                            message=(
                                f"Possible duplicate: '{entities[indices[i]]}' and "
                                f"'{entities[indices[j]]}' have very similar embeddings "
                                f"(dist={dist:.4f}) — may refer to the same entity"
                            ),
                            node_label=label,
                        ))

        outlier_count = sum(1 for v in violations if v.severity == "warning")
        score = max(0.0, 1.0 - (outlier_count * 0.15))

        return DimensionResult(
            dimension="entity_clustering",
            score=min(1.0, score),
            violations=violations,
            details={
                "entity_count": len(entities),
                "label_groups": {k: len(v) for k, v in label_groups.items()},
                "outlier_count": outlier_count,
                "duplicate_candidates": sum(1 for v in violations if v.severity == "info"),
                "model": "ComplEx",
            },
        )

    except Exception as e:
        return DimensionResult(
            dimension="entity_clustering",
            score=0.5,
            violations=[Violation(
                dimension="entity_clustering",
                severity="info",
                message=f"Entity clustering failed: {e}",
            )],
            details={"error": str(e)},
        )
