"""Set, logic, relation, and function utilities for SRAI notebooks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Hashable


def union(a: Iterable[Hashable], b: Iterable[Hashable]) -> set[Hashable]:
    """Return the union of two finite collections."""
    return set(a) | set(b)


def intersection(a: Iterable[Hashable], b: Iterable[Hashable]) -> set[Hashable]:
    """Return the intersection of two finite collections."""
    return set(a) & set(b)


def difference(a: Iterable[Hashable], b: Iterable[Hashable]) -> set[Hashable]:
    """Return the elements in a that are not in b."""
    return set(a) - set(b)


def complement(a: Iterable[Hashable], universe: Iterable[Hashable]) -> set[Hashable]:
    """Return the complement of a relative to a finite universe."""
    a_set = set(a)
    universe_set = set(universe)
    if not a_set <= universe_set:
        raise ValueError("The set must be a subset of the universe.")
    return universe_set - a_set


def cartesian_product(
    a: Iterable[Hashable], b: Iterable[Hashable]
) -> set[tuple[Hashable, Hashable]]:
    """Return the Cartesian product A × B."""
    return {(x, y) for x in set(a) for y in set(b)}


def implies(p: bool, q: bool) -> bool:
    """Return the material implication p -> q."""
    return (not p) or q


def equivalent(p: bool, q: bool) -> bool:
    """Return logical equivalence p <-> q."""
    return p == q


def truth_table_2(
    operation: Callable[[bool, bool], bool],
) -> list[dict[str, bool]]:
    """Generate a two-variable truth table for a Boolean operation."""
    rows: list[dict[str, bool]] = []
    for p in (True, False):
        for q in (True, False):
            rows.append({"P": p, "Q": q, "Result": bool(operation(p, q))})
    return rows


Relation = set[tuple[Hashable, Hashable]]


def relation_domain(relation: Relation) -> set[Hashable]:
    """Return the domain of a finite binary relation."""
    return {x for x, _ in relation}


def relation_range(relation: Relation) -> set[Hashable]:
    """Return the range of a finite binary relation."""
    return {y for _, y in relation}


def is_function(relation: Relation) -> bool:
    """Return True if every input in the relation maps to exactly one output."""
    mapping: dict[Hashable, Hashable] = {}
    for x, y in relation:
        if x in mapping and mapping[x] != y:
            return False
        mapping[x] = y
    return True


def evaluate_predicate(
    values: Iterable[Any], predicate: Callable[[Any], bool]
) -> set[Any]:
    """Return the subset of values satisfying a predicate."""
    return {value for value in values if predicate(value)}
