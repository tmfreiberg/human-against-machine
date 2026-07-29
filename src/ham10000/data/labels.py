"""Construction of label schemes for the classification tasks.

HAM10000 ships seven diagnostic classes. The project runs several tasks over
them: the native seven-way problem, reduced groupings such as
malignant/benign, and binary melanoma-versus-nevus. This module turns a task
specification into the two mappings the rest of the pipeline needs.

The logic is pure functions over plain data rather than methods on a pipeline
object, so a scheme can be built and inspected in a test without touching a
dataset, and a bad specification raises where it is made rather than failing
later in something unrelated.

Two conventions are preserved exactly, because every stored artefact --
probability CSVs with `prob_<name>` columns, saved state dicts whose final
layer width equals the class count -- depends on them:

* **Sorted order.** Class names are sorted before indices are assigned, so the
  same specification always yields the same integer for the same class. Without
  this, a model trained in one session is silently mismatched with predictions
  decoded in another.
* **`other` takes index 0.** When a specification does not cover every
  diagnosis present in the data, the remainder is collected into a class named
  `other`, which is always index 0 and shifts the rest up by one. When the
  specification is exhaustive, there is no `other` and indexing starts at 0
  with the first sorted class.

Examples
--------
A binary task over a dataset that also contains other diagnoses:

>>> scheme = LabelScheme.from_diagnoses(["mel", "nv"], present={"mel", "nv", "bkl"})
>>> scheme.codes
{0: 'other', 1: 'mel', 2: 'nv'}
>>> scheme.mapping["bkl"]
0

An exhaustive specification has no `other`:

>>> LabelScheme.from_diagnoses(["mel", "nv"], present={"mel", "nv"}).codes
{0: 'mel', 1: 'nv'}
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

__all__ = ["OTHER", "LabelScheme"]

#: Name of the catch-all class for diagnoses outside the specification.
OTHER = "other"


@dataclass(frozen=True, slots=True)
class LabelScheme:
    """A bidirectional mapping between diagnoses and integer class labels.

    Parameters
    ----------
    codes:
        Integer index to class name. This is the order a model's output layer
        is in, so `codes[i]` names column `i` of a probability matrix.
    mapping:
        Diagnosis string to integer index. Several diagnoses may share an
        index when the task groups them.

    Examples
    --------
    >>> scheme = LabelScheme(codes={0: "nv", 1: "mel"}, mapping={"nv": 0, "mel": 1})
    >>> scheme.n_classes
    2
    >>> scheme.name_of(1)
    'mel'
    >>> scheme.probability_columns
    ['prob_nv', 'prob_mel']
    """

    codes: Mapping[int, str]
    mapping: Mapping[str, int]

    @property
    def n_classes(self) -> int:
        """Number of distinct classes, i.e. the width of the model's head."""
        return len(self.codes)

    @property
    def probability_columns(self) -> list[str]:
        """Column names for a probability table, in model output order.

        The `prob_` prefix and the ordering are load-bearing: stored
        probability CSVs use these names, and downstream aggregation selects
        them with a regular expression.

        Examples
        --------
        >>> LabelScheme({0: "a", 1: "b"}, {"a": 0, "b": 1}).probability_columns
        ['prob_a', 'prob_b']
        """
        return [f"prob_{self.codes[i]}" for i in sorted(self.codes)]

    def name_of(self, index: int) -> str:
        """Return the class name for an integer index.

        Raises
        ------
        KeyError
            If the index is not part of this scheme. Returning the index
            itself on a miss would produce mixed-type output downstream, where a
            column of class names silently gains integers.

        Examples
        --------
        >>> LabelScheme({0: "nv"}, {"nv": 0}).name_of(0)
        'nv'
        """
        return self.codes[index]

    @classmethod
    def from_diagnoses(
        cls,
        to_classify: Sequence[str],
        present: Collection[str],
    ) -> LabelScheme:
        """Build a scheme that treats each named diagnosis as its own class.

        Parameters
        ----------
        to_classify:
            Diagnoses to model individually. Names not present in the data are
            dropped rather than raising.
        present:
            Diagnoses actually occurring in the dataset.

        Returns
        -------
        LabelScheme

        Raises
        ------
        ValueError
            If no requested diagnosis occurs in the data. An empty scheme
            would fail later with an error unrelated to its cause.

        Examples
        --------
        >>> LabelScheme.from_diagnoses(["nv", "mel"], {"nv", "mel", "bcc"}).codes
        {0: 'other', 1: 'mel', 2: 'nv'}

        Requested classes absent from the data are dropped:

        >>> LabelScheme.from_diagnoses(["mel", "unicorn"], {"mel", "nv"}).codes
        {0: 'other', 1: 'mel'}
        """
        present_set = set(present)
        wanted = sorted(present_set.intersection(to_classify))
        if not wanted:
            raise ValueError(
                f"None of the requested diagnoses {sorted(to_classify)} occur in "
                f"the data, which contains {sorted(present_set)}."
            )

        remainder = sorted(present_set - set(wanted))
        if remainder:
            codes = {0: OTHER} | {i + 1: dx for i, dx in enumerate(wanted)}
            mapping = dict.fromkeys(remainder, 0)
            mapping |= {dx: i + 1 for i, dx in enumerate(wanted)}
        else:
            codes = dict(enumerate(wanted))
            mapping = {dx: i for i, dx in enumerate(wanted)}

        return cls(codes=codes, mapping=mapping)

    @classmethod
    def from_groups(
        cls,
        to_classify: Mapping[str, Sequence[str]],
        present: Collection[str],
    ) -> LabelScheme:
        """Build a scheme that collapses groups of diagnoses into single classes.

        This is how the malignant/benign task is expressed.

        Parameters
        ----------
        to_classify:
            Class name to the diagnoses it covers. Diagnoses absent from the
            data are dropped, and a group left empty is discarded.
        present:
            Diagnoses actually occurring in the dataset.

        Returns
        -------
        LabelScheme

        Raises
        ------
        ValueError
            If a diagnosis appears in more than one group, or if no group
            survives filtering.

        Notes
        -----
        Groups must be disjoint. Letting a later group silently win would mean
        a specification with `mel` in both `malignant` and `high_risk` trains
        against a mapping the caller did not intend, with nothing to reveal it.

        Examples
        --------
        >>> scheme = LabelScheme.from_groups(
        ...     {"malignant": ["mel", "bcc"], "benign": ["nv", "bkl"]},
        ...     present={"mel", "bcc", "nv", "bkl"},
        ... )
        >>> scheme.codes
        {0: 'benign', 1: 'malignant'}
        >>> scheme.mapping["mel"], scheme.mapping["nv"]
        (1, 0)

        Uncovered diagnoses become `other` at index 0, shifting the rest up:

        >>> scheme = LabelScheme.from_groups(
        ...     {"malignant": ["mel"], "benign": ["nv"]},
        ...     present={"mel", "nv", "vasc"},
        ... )
        >>> scheme.codes
        {0: 'other', 1: 'benign', 2: 'malignant'}
        """
        present_set = set(present)

        groups: dict[str, list[str]] = {}
        for name, diagnoses in to_classify.items():
            kept = sorted(present_set.intersection(diagnoses))
            if kept:
                groups[name] = kept

        if not groups:
            raise ValueError(
                "No group contains a diagnosis present in the data "
                f"({sorted(present_set)})."
            )

        seen: dict[str, str] = {}
        for name, diagnoses in groups.items():
            for dx in diagnoses:
                if dx in seen:
                    raise ValueError(
                        f"Diagnosis {dx!r} appears in both {seen[dx]!r} and "
                        f"{name!r}. Groups must be disjoint."
                    )
                seen[dx] = name

        ordered = sorted(groups)
        covered = set(seen)
        remainder = sorted(present_set - covered)

        if remainder:
            codes = {0: OTHER} | {i + 1: name for i, name in enumerate(ordered)}
            groups[OTHER] = remainder
        else:
            codes = dict(enumerate(ordered))

        mapping = {dx: index for index, name in codes.items() for dx in groups[name]}
        return cls(codes=codes, mapping=mapping)

    @classmethod
    def build(
        cls,
        to_classify: Sequence[str] | Mapping[str, Sequence[str]],
        present: Collection[str],
    ) -> LabelScheme:
        """Dispatch to :meth:`from_groups` or :meth:`from_diagnoses`.

        Convenience for callers holding a specification that may be either a
        list of diagnoses or a mapping of groups.

        Examples
        --------
        >>> LabelScheme.build(["mel", "nv"], {"mel", "nv"}).n_classes
        2
        >>> LabelScheme.build({"m": ["mel"], "b": ["nv"]}, {"mel", "nv"}).n_classes
        2
        """
        if isinstance(to_classify, Mapping):
            return cls.from_groups(to_classify, present)
        return cls.from_diagnoses(to_classify, present)
