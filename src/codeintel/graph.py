"""In-memory code graph over language-neutral Symbols and Relations."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from enum import StrEnum

from codeintel.models import Relation, RelationKind, ResolutionStatus, Symbol

_FOLLOWABLE = frozenset({ResolutionStatus.RESOLVED, ResolutionStatus.PROBABLE})


class TraversalDirection(StrEnum):
    """Direction used when walking graph adjacency."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class CodeGraph:
    """Directed graph whose nodes are repository Symbols identified by qualified_name."""

    def __init__(self, symbols: Iterable[Symbol], relations: Iterable[Relation]) -> None:
        self._symbols: dict[str, Symbol] = {}
        for symbol in symbols:
            existing = self._symbols.get(symbol.qualified_name)
            if existing is not None and existing is not symbol:
                raise ValueError(f"Duplicate qualified name: {symbol.qualified_name}")
            self._symbols[symbol.qualified_name] = symbol
        self._relations = tuple(_sorted_relations(relations))
        outgoing: dict[str, list[Relation]] = defaultdict(list)
        incoming: dict[str, list[Relation]] = defaultdict(list)
        for relation in self._relations:
            outgoing[relation.source_qualified_name].append(relation)
            if (
                relation.resolution in _FOLLOWABLE
                and relation.target_qualified_name is not None
                and relation.target_qualified_name in self._symbols
            ):
                incoming[relation.target_qualified_name].append(relation)
        self._outgoing = {key: tuple(_sorted_relations(value)) for key, value in outgoing.items()}
        self._incoming = {key: tuple(_sorted_relations(value)) for key, value in incoming.items()}

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        return tuple(self._symbols[name] for name in sorted(self._symbols))

    @property
    def relations(self) -> tuple[Relation, ...]:
        return self._relations

    def get_symbol(self, qualified_name: str) -> Symbol:
        """Return the Symbol for ``qualified_name``.

        Raises:
            KeyError: if the name is not a graph node.
        """
        try:
            return self._symbols[qualified_name]
        except KeyError:
            raise KeyError(f"Unknown symbol: {qualified_name}") from None

    def has_symbol(self, qualified_name: str) -> bool:
        return qualified_name in self._symbols

    def outgoing(
        self,
        qualified_name: str,
        *,
        kinds: Sequence[RelationKind] | None = None,
        resolutions: Sequence[ResolutionStatus] | None = None,
    ) -> tuple[Relation, ...]:
        self.get_symbol(qualified_name)
        return _filter_relations(self._outgoing.get(qualified_name, ()), kinds, resolutions)

    def incoming(
        self,
        qualified_name: str,
        *,
        kinds: Sequence[RelationKind] | None = None,
        resolutions: Sequence[ResolutionStatus] | None = None,
    ) -> tuple[Relation, ...]:
        self.get_symbol(qualified_name)
        return _filter_relations(self._incoming.get(qualified_name, ()), kinds, resolutions)

    def neighbors(
        self,
        qualified_name: str,
        *,
        direction: TraversalDirection = TraversalDirection.OUTGOING,
        kinds: Sequence[RelationKind] | None = None,
        resolutions: Sequence[ResolutionStatus] | None = None,
    ) -> tuple[Symbol, ...]:
        names: set[str] = set()
        for relation in self._followable_edges(qualified_name, direction, kinds, resolutions):
            neighbor = _other_end(qualified_name, relation)
            if neighbor is not None:
                names.add(neighbor)
        return tuple(self._symbols[name] for name in sorted(names) if name in self._symbols)

    def bounded_neighborhood(
        self,
        qualified_name: str,
        *,
        max_depth: int,
        direction: TraversalDirection = TraversalDirection.OUTGOING,
        kinds: Sequence[RelationKind] | None = None,
        resolutions: Sequence[ResolutionStatus] | None = None,
    ) -> tuple[tuple[str, int], ...]:
        """Return ``(qualified_name, distance)`` pairs within ``max_depth`` via BFS."""
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        self.get_symbol(qualified_name)
        distances = self._bfs(
            qualified_name,
            max_depth=max_depth,
            direction=direction,
            kinds=kinds,
            resolutions=resolutions,
        )
        return tuple(sorted(distances.items(), key=lambda item: (item[1], item[0])))

    def shortest_distance(
        self,
        source: str,
        target: str,
        *,
        max_depth: int | None = None,
        kinds: Sequence[RelationKind] | None = None,
        resolutions: Sequence[ResolutionStatus] | None = None,
    ) -> int | None:
        self.get_symbol(source)
        self.get_symbol(target)
        distances = self._bfs(
            source,
            max_depth=max_depth,
            direction=TraversalDirection.OUTGOING,
            kinds=kinds,
            resolutions=resolutions,
        )
        return distances.get(target)

    def _followable_edges(
        self,
        qualified_name: str,
        direction: TraversalDirection,
        kinds: Sequence[RelationKind] | None,
        resolutions: Sequence[ResolutionStatus] | None,
    ) -> tuple[Relation, ...]:
        relations: list[Relation] = []
        if direction in {TraversalDirection.OUTGOING, TraversalDirection.BOTH}:
            relations.extend(self.outgoing(qualified_name, kinds=kinds, resolutions=resolutions))
        if direction in {TraversalDirection.INCOMING, TraversalDirection.BOTH}:
            relations.extend(self.incoming(qualified_name, kinds=kinds, resolutions=resolutions))
        followable: list[Relation] = []
        seen: set[tuple[object, ...]] = set()
        for relation in relations:
            if relation.target_qualified_name is None:
                continue
            if relation.resolution not in _FOLLOWABLE:
                continue
            key = (
                relation.kind,
                relation.source_qualified_name,
                relation.target_qualified_name,
                relation.target_text,
                str(relation.path),
            )
            if key in seen:
                continue
            seen.add(key)
            followable.append(relation)
        return tuple(followable)

    def _bfs(
        self,
        start: str,
        *,
        max_depth: int | None,
        direction: TraversalDirection,
        kinds: Sequence[RelationKind] | None,
        resolutions: Sequence[ResolutionStatus] | None,
    ) -> dict[str, int]:
        distances = {start: 0}
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            current_depth = distances[current]
            if max_depth is not None and current_depth >= max_depth:
                continue
            for relation in self._followable_edges(current, direction, kinds, resolutions):
                neighbor = _other_end(current, relation)
                if neighbor is None or neighbor not in self._symbols:
                    continue
                if neighbor not in distances:
                    distances[neighbor] = current_depth + 1
                    queue.append(neighbor)
        return distances


def _other_end(qualified_name: str, relation: Relation) -> str | None:
    if relation.source_qualified_name == qualified_name:
        return relation.target_qualified_name
    if relation.target_qualified_name == qualified_name:
        return relation.source_qualified_name
    return None


def _filter_relations(
    relations: Sequence[Relation],
    kinds: Sequence[RelationKind] | None,
    resolutions: Sequence[ResolutionStatus] | None,
) -> tuple[Relation, ...]:
    kind_set = frozenset(kinds) if kinds is not None else None
    resolution_set = frozenset(resolutions) if resolutions is not None else None
    selected = []
    for relation in relations:
        if kind_set is not None and relation.kind not in kind_set:
            continue
        if resolution_set is not None and relation.resolution not in resolution_set:
            continue
        selected.append(relation)
    return tuple(selected)


def _sorted_relations(relations: Iterable[Relation]) -> tuple[Relation, ...]:
    return tuple(
        sorted(
            relations,
            key=lambda relation: (
                relation.kind.value,
                relation.source_qualified_name,
                relation.target_text,
                relation.target_qualified_name or "",
                str(relation.path),
                relation.span.start_byte if relation.span is not None else -1,
            ),
        )
    )
