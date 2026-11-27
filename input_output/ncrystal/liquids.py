"""I/O support for the external NCrystal Liquids plugin's NCMAT extension.

The custom-liquid block configures *NCrystal's convolution physics*, not an
extra VDOS. All file modifications preserve unrelated NCMAT sections verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Iterable

import numpy as np

from common.validation import (
    require_finite_array,
    require_finite_positive,
    require_ndim,
    require_nonnegative_array,
    require_same_shape,
)


@dataclass(frozen=True)
class LiquidAtomSettings:
    """Settings for one ``@CUSTOM_LIQUIDS`` entry.

    Parameters
    ----------
    atom : str
        NCMAT atom label, e.g. ``H``.
    model : str
        ``incoherent``, ``skold`` or ``vineyard``.
    c : float
        NCrystal Liquids dimensionless Egelstaff--Schofield parameter. Do not
        blindly equate it with the parameter of an unrelated diffusion model.
    wt, ws : float
        Translational and solid-like mode weights. Their sum must be unity.
    static_structure_factor : tuple of tuple of float, optional
        Ordered pairs of momentum transfer Q (inverse ångströms) and
        dimensionless S(Q). Required for Sköld/Vineyard models, forbidden
        for incoherent mode. Defaults to an empty tuple.
    yk_model : str or None, optional
        Young--Koppel spin-state option, ``"ortho"`` or ``"para"``.
        Supported only for H or D. Defaults to None.

    Notes
    -----
    ``c`` must be finite and positive. ``wt`` and ``ws`` must each lie in
    [0, 1], and their sum must equal one within an absolute tolerance of
    1e-5. Non-incoherent models require at least two strictly increasing
    Q values with positive S(Q).
    """

    # Species and translation model.
    atom: str
    model: str

    # Diffusion parameter and component weights.
    c: float
    wt: float
    ws: float

    # Liquid structure and kernel options.
    static_structure_factor: tuple[tuple[float, float], ...] = ()
    yk_model: str | None = None

    def __post_init__(self) -> None:
        """Validate liquid plugin model, weights, and structure factor settings.

        Raises
        ------
        ValueError
            If the atom label or model is unsupported, ``c`` is not finite
            and positive, the component weights violate their bounds or sum
            constraint, the Young--Koppel setting is incompatible with the
            species, or the S(Q) data violate the selected model's constraints.
        """
        self._validate_translation_settings()
        self._validate_structure_factor()

    def _validate_translation_settings(self) -> None:
        """Validate the plugin species, translation parameters, and mode weights.

        Raises
        ------
        ValueError
            If the species, model, translation parameter, mode weights, or
            Young--Koppel setting are invalid.
        """
        # The plugin identifies species by chemical element label.
        if not re.fullmatch(r"[A-Z][a-z]?|D", self.atom):
            raise ValueError("atom must be an element label (H, O, D, ...).")
        if self.model not in ("incoherent", "skold", "vineyard"):
            raise ValueError("Liquid plugin model must be incoherent, skold, or vineyard.")
        # The Egelstaff--Schofield parameter must be physical for the plugin.
        require_finite_positive(self.c, message="Liquid plugin c must be finite and positive.")
        # Translation and solid-like contributions partition the mode weight.
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (self.wt, self.ws)):
            raise ValueError("Liquid mode weights must lie in [0,1].")
        # Accept rounded CAB weights, but do not renormalize them silently.
        if not math.isclose(self.wt + self.ws, 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("Liquid plugin wt + ws must equal 1.")
        # Young--Koppel ortho/para options apply only to hydrogen isotopes.
        if self.yk_model is not None:
            if self.atom not in ("H", "D") or self.yk_model not in ("ortho", "para"):
                raise ValueError("yk_model ortho/para is only valid for H or D.")

    def _validate_structure_factor(self) -> None:
        """Validate structure-factor rows for the selected scattering model.

        Raises
        ------
        ValueError
            If S(Q) presence, length, ordering, or values violate model constraints.
        """
        sq = self.static_structure_factor
        # Coherent approximations need tabulated S(Q); incoherent mode forbids it.
        if self.model == "incoherent" and sq:
            raise ValueError("Incoherent mode must not include S(Q) rows.")
        if self.model != "incoherent" and len(sq) < 2:
            raise ValueError("Skold/Vineyard mode requires at least 2 measured S(Q) rows.")
        # Strict Q ordering avoids ambiguous interpolation in the native plugin.
        previous = -1.0
        for q, s in sq:
            if not all(math.isfinite(x) for x in (q, s)) or q < 0 or q <= previous or s <= 0:
                raise ValueError("S(Q) rows need ascending Q >= 0 and positive S(Q).")
            previous = q


def render_custom_liquids(entries: Iterable[LiquidAtomSettings]) -> str:
    """Serialize validated plugin settings using the upstream README syntax.

    Parameters
    ----------
    entries : Iterable[LiquidAtomSettings]
        Validated atom-specific Liquids plugin settings.

    Returns
    -------
    str
        Serialized ``@CUSTOM_LIQUIDS`` section, including its header and
        trailing newline.

    Raises
    ------
    ValueError
        If no entries are supplied or an atom label appears more than once.
    """
    # Materialize once so generators can be checked for duplicate species.
    entries = tuple(entries)
    if not entries or len({entry.atom for entry in entries}) != len(entries):
        raise ValueError("At least one unique atom entry is needed.")

    # Keep each species header adjacent to its optional spin and S(Q) rows.
    lines = ["@CUSTOM_LIQUIDS"]
    for item in entries:
        lines.append(f"  {item.atom} {item.model} {item.c:.12g} {item.wt:.12g} {item.ws:.12g}")
        if item.yk_model is not None:
            lines.append(f"  yk_model {item.yk_model}")
        lines.extend(f"  {q:.12g} {s:.12g}" for q, s in item.static_structure_factor)
    return "\n".join(lines) + "\n"


def _split_ncmat(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split NCMAT into header and tagged raw sections without restructuring.

    Parameters
    ----------
    text : str
        Original NCMAT document text.

    Returns
    -------
    tuple[str, list[tuple[str, str]]]
        Unmodified header text and ordered ``(section_tag, section_text)``
        pairs. Whitespace and unrecognized sections are preserved.

    Raises
    ------
    ValueError
        If the document lacks an NCMAT version header or tagged sections.
    """
    # Refuse non-NCMAT input before manipulating raw section boundaries.
    if not re.match(r"^NCMAT v\d+\s*(?:\n|$)", text):
        raise ValueError("Expected an NCMAT version header.")

    # Record section offsets rather than parsing and rewriting unrelated data.
    boundaries = list(re.finditer(r"(?m)^@([A-Z][A-Z0-9_]*)\s*$", text))
    if not boundaries:
        raise ValueError("No NCMAT sections found.")

    # Retain original whitespace and section ordering for lossless round-trips.
    sections = []
    for i, match in enumerate(boundaries):
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        sections.append((match.group(1), text[match.start():end]))
    return text[:boundaries[0].start()], sections


def ensure_custom_liquids(text: str) -> None:
    """Validate that an NCMAT document has one populated liquid extension.

    Parameters
    ----------
    text : str
        Complete NCMAT document to inspect.

    Returns
    -------
    None
        Returns normally only when the block and its entries are valid.

    Raises
    ------
    ValueError
        If NCMAT framing is invalid, there is not exactly one populated
        ``@CUSTOM_LIQUIDS`` block, or the block entries are invalid.
    """
    _, sections = _split_ncmat(text)

    # Exactly one populated custom block is required for liquid semantics.
    blocks = [data for name, data in sections if name == "CUSTOM_LIQUIDS"]
    if len(blocks) != 1 or not any(line.strip() and not line.lstrip().startswith('#')
                                  for line in blocks[0].splitlines()[1:]):
        raise ValueError("A single populated @CUSTOM_LIQUIDS block is required.")
    _parse_custom_liquids_block(blocks[0])


def _parse_liquid_atom_rows(data: list[list[str]], i: int) -> tuple[LiquidAtomSettings, int]:
    """Read one species and its optional subordinate rows.

    Parameters
    ----------
    data : list of list of str
        Tokenized plugin rows without comments.
    i : int
        Index of the species header.

    Returns
    -------
    tuple of LiquidAtomSettings and int
        Validated species data and index of the next header.

    Raises
    ------
    ValueError
        If the species header or its subordinate fields are malformed.
    """
    # Each atom starts with five tokens, followed by optional subordinate rows.
    head = data[i]
    if len(head) != 5 or head[0] == 'yk_model':
        raise ValueError('Expected atom model c wt ws in @CUSTOM_LIQUIDS.')
    atom, mode = head[:2]
    try:
        c, wt, ws = map(float, head[2:])
    except ValueError as exc:
        raise ValueError('Invalid liquid model parameters.') from exc
    i += 1

    # Spin-state data, when supplied, belongs to the immediately prior atom.
    yk = None
    if i < len(data) and data[i][0] == 'yk_model':
        if len(data[i]) != 2:
            raise ValueError('Expected yk_model ortho or para.')
        yk = data[i][1]
        i += 1

    # A run of numeric pairs belongs to this atom until the next header.
    sq = []
    while i < len(data) and len(data[i]) == 2 and data[i][0] != 'yk_model':
        try:
            q, s = float(data[i][0]), float(data[i][1])
        except ValueError:
            break
        sq.append((q, s))
        i += 1

    # Delegate physical and model-specific constraints to the data class.
    return LiquidAtomSettings(atom, mode, c, wt, ws, tuple(sq), yk), i


def _parse_custom_liquids_block(block: str) -> tuple[LiquidAtomSettings, ...]:
    """Validate every plugin row, including multi-atom and Young--Koppel data.

    Parameters
    ----------
    block : str
        Text of a single CUSTOM_LIQUIDS section.

    Returns
    -------
    tuple[LiquidAtomSettings, ...]
        Validated atom-specific settings in their original order.

    Raises
    ------
    ValueError
        If a header or numeric value is malformed, entries are duplicated,
        or a setting violates ``LiquidAtomSettings`` validation.
    """
    # Ignore blank lines and trailing comments without changing numeric tokens.
    data = []
    for raw in block.splitlines()[1:]:
        parts = raw.split('#', 1)[0].split()
        if parts:
            data.append(parts)
    entries = []
    i = 0
    while i < len(data):
        entry, i = _parse_liquid_atom_rows(data, i)
        entries.append(entry)
    if len(set(e.atom for e in entries)) != len(entries):
        raise ValueError('Duplicate atom entries in @CUSTOM_LIQUIDS.')
    return tuple(entries)


def parse_custom_liquids(text: str) -> tuple[LiquidAtomSettings, ...]:
    """Parse the unique liquid plugin section of an NCMAT document.

    Parameters
    ----------
    text : str
        NCMAT document containing an ``@CUSTOM_LIQUIDS`` section.

    Returns
    -------
    tuple of LiquidAtomSettings
        Validated settings for each atom, in document order.

    Raises
    ------
    ValueError
        If NCMAT structure is invalid, the plugin section is missing or
        duplicated, or any plugin entry fails validation.
    """
    _, sections = _split_ncmat(text)

    # A duplicate block is ambiguous even if both blocks look individually valid.
    blocks = [body for name, body in sections if name == 'CUSTOM_LIQUIDS']
    if len(blocks) != 1:
        raise ValueError('Expected exactly one @CUSTOM_LIQUIDS block.')
    return _parse_custom_liquids_block(blocks[0])


def add_custom_liquids(text: str, entries: Iterable[LiquidAtomSettings]) -> str:
    """Append a new liquid plugin section to an NCMAT document.

    Parameters
    ----------
    text : str
        Base NCMAT document with no existing ``@CUSTOM_LIQUIDS`` section.
    entries : Iterable[LiquidAtomSettings]
        Validated settings for one or more distinct species.

    Returns
    -------
    str
        Source document with a serialized liquid section appended.

    Raises
    ------
    ValueError
        If the source has invalid NCMAT framing, already contains a liquid
        section, or the settings are empty or contain duplicate species.

    Notes
    -----
    This function does not replace an existing liquid section. It trims
    trailing whitespace from the input before appending the new block.
    """
    _, sections = _split_ncmat(text)

    # Refuse implicit replacement: callers must explicitly edit existing models.
    if any(tag == "CUSTOM_LIQUIDS" for tag, _ in sections):
        raise ValueError("The source already has @CUSTOM_LIQUIDS; edit it explicitly.")
    return text.rstrip() + "\n\n" + render_custom_liquids(entries)


def replace_custom_liquids(text: str, entries: Iterable[LiquidAtomSettings]) -> str:
    """Replace an existing liquid section and preserve the other NCMAT sections.

    Parameters
    ----------
    text : str
        NCMAT document with exactly one custom liquid section.
    entries : Iterable[LiquidAtomSettings]
        Replacement settings for the liquid species.

    Returns
    -------
    str
        NCMAT document containing the replacement section.
    """
    header, sections = _split_ncmat(text)
    if sum(tag == "CUSTOM_LIQUIDS" for tag, _ in sections) != 1:
        raise ValueError("Expected exactly one @CUSTOM_LIQUIDS block.")

    # Replace the custom block in place so section order and unrelated material text are
    # preserved.
    liquid = render_custom_liquids(entries)
    return header + "".join(liquid if tag == "CUSTOM_LIQUIDS" else raw
                            for tag, raw in sections)


def validate_regular_vdos_grid(energy_ev) -> float:
    """Validate an NCrystal VDOS grid without invoking its C++ regulariser.

    In the NCrystal v7 VDOS format E_min must be a positive integer multiple
    of the uniform bin width. The inappropriate 1e-5..0.423 eV grid (1201
    points) is *not* valid: it caused the native ``mm <= 20000`` assertion.
    Return the regular grid spacing in eV; never silently shift a spectrum.

    Parameters
    ----------
    energy_ev : array_like
        One-dimensional VDOS energy coordinates in electronvolts, with at
        least three finite, strictly increasing values.

    Returns
    -------
    float
        Uniform grid spacing in electronvolts.

    Raises
    ------
    ValueError
        If the grid is not one-dimensional, contains fewer than three
        finite points, is not positive and increasing, has nonuniform
        spacing, or has an initial energy not aligned to an integer
        multiple of the spacing within the implemented tolerances.

    Notes
    -----
    Uniformity uses ``rtol=1e-6`` and ``atol=1e-12``; alignment uses a
    ratio tolerance of 1e-6. No resampling or shifting is performed.
    """
    # Work on numeric grid values without altering the supplied spectrum.
    grid = np.asarray(energy_ev, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or not np.isfinite(grid).all():
        raise ValueError("NCrystal VDOS energy grid must contain >=3 finite values.")

    # NCMAT v7 stores only grid endpoints and assumes a constant interval.
    diffs = np.diff(grid)
    spacing = float((grid[-1] - grid[0]) / (grid.size - 1))
    if grid[0] < 1e-5 or spacing <= 0 or not np.all(diffs > 0):
        raise ValueError("NCrystal VDOS energy grid must be positive and increasing.")
    if not np.allclose(diffs, spacing, rtol=1e-6, atol=1e-12):
        raise ValueError("NCrystal VDOS energy grid must be uniformly spaced.")

    # The native VDOS regularizer requires E_min / delta_E to be integral.
    # Reject misalignment instead of shifting data and changing its physics.
    ratio = float(grid[0] / spacing)
    if ratio < 1 - 1e-6 or abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(
            "NCrystal VDOS energy grid is not aligned: E_min / delta_E must "
            f"be a positive integer, got {ratio:.9g}. "
            "Build a regular grid in the material adapter; do not silently "
            "shift or rescale it in the NCMAT serializer."
        )
    return spacing


def _validate_vdos_input(energy_ev, density) -> tuple[np.ndarray, np.ndarray]:
    """Check that a candidate VDOS is safe to serialize.

    Parameters
    ----------
    energy_ev : array_like
        Target VDOS energy grid in eV.
    density : array_like
        Finite, nonnegative VDOS density samples matching ``energy_ev``
        element-for-element. The values are not normalized here.

    Returns
    -------
    tuple of numpy.ndarray
        One-dimensional floating-point arrays ``(energy_grid, density)``
        with identical shapes and at least three points.

    Raises
    ------
    ValueError
        If shapes differ, input values are nonfinite or negative where
        forbidden, or the energy grid fails regular-grid validation.
    """
    # Validate grid and density together before any NCMAT serialization.
    grid = np.asarray(energy_ev, dtype=float)
    vals = np.asarray(density, dtype=float)
    grid_error = "VDOS grid and density must be matching 1D arrays (>=3)."
    require_ndim(grid, 1, message=grid_error)
    require_same_shape(vals, grid, message=grid_error)
    if len(grid) < 3:
        raise ValueError(grid_error)
    density_error = "VDOS densities must be finite and nonnegative."
    require_finite_array(vals, message=density_error)
    require_nonnegative_array(vals, message=density_error)
    validate_regular_vdos_grid(grid)
    return grid, vals


def _preserved_dyninfo_lines(body: str) -> list[str]:
    """Retain recognized DYNINFO metadata while rejecting unknown fields.

    Parameters
    ----------
    body : str
        Original ``@DYNINFO`` section.

    Returns
    -------
    list of str
        Original comment lines and supported dynamic-information metadata
        lines, preserving their source text and ordering.

    Raises
    ------
    ValueError
        If a nonnumeric, unsupported DYNINFO field is encountered.

    Notes
    -----
    Original VDOS fields and numeric continuation rows are intentionally
    omitted because the caller writes a replacement spectrum.
    """
    # Preserve original metadata/comment lines; old VDOS samples are replaced.
    preserve = []
    for line in body.splitlines()[1:]:
        token = line.strip().split()
        if not token or line.lstrip().startswith('#'):
            if line.lstrip().startswith('#'):
                preserve.append(line)
            continue
        if token[0] in ("element", "fraction", "type", "debye_temp"):
            preserve.append(line)
        elif token[0] not in ("vdos_egrid", "vdos", "vdos_density"):
            # Numeric continuation lines belong to the old VDOS payload.
            # Continuation lines may contain only numeric values.
            try:
                [float(value) for value in token]
            except ValueError as exc:
                raise ValueError(f"Unsupported DYNINFO field: {token[0]}") from exc
    return preserve


def _render_vdos_dyninfo(body: str, grid: np.ndarray, vals: np.ndarray) -> str:
    """Render the target DYNINFO block without changing unrelated metadata.

    Parameters
    ----------
    body : str
        Original dynamic-information section.
    grid : numpy.ndarray
        Validated regular energy grid in eV.
    vals : numpy.ndarray
        Validated VDOS density values corresponding to ``grid``.

    Returns
    -------
    str
        Replacement NCMAT ``@DYNINFO`` section using the two-endpoint
        energy-grid representation and density values formatted to 12
        significant digits.

    Raises
    ------
    ValueError
        If the source section contains unsupported nonnumeric fields.

    Notes
    -----
    This serializer assumes ``grid`` and ``vals`` have already passed
    ``_validate_vdos_input``; it does not normalize the density.
    """
    # Retain atom metadata but substitute one regularized-grid representation.
    out = ["@DYNINFO", *_preserved_dyninfo_lines(body),
           f"  vdos_egrid {grid[0]:.15g} {grid[-1]:.15g}"]

    # NCMAT v7 requires the first density values on the keyword line.
    for index in range(0, len(vals), 6):
        values = " ".join(f"{value:.12g}" for value in vals[index:index + 6])
        out.append(("  vdos_density " if index == 0 else "    ") + values)
    return "\n".join(out) + "\n\n"


def replace_vdos(text: str, atom: str, energy_ev, density) -> str:
    """Replace one species' VDOS without removing liquid plugin settings.

    A uniform positive energy grid is required for the NCMAT ``vdos_egrid``
    two-endpoint representation. The supplied density must already include
    any intramolecular oscillator modes; this function does not normalize it.

    Parameters
    ----------
    text : str
        Original NCMAT document.
    atom : str
        Species element label matching exactly one ``@DYNINFO`` section.
    energy_ev : array_like
        NCrystal-compatible energy grid in eV.
    density : array_like
        Finite, nonnegative VDOS density on the supplied grid. No
        normalization or oscillator processing is performed here.

    Returns
    -------
    str
        Updated NCMAT document with unrelated sections preserved.

    Raises
    ------
    ValueError
        If the input VDOS or NCMAT structure is invalid, the liquid plugin
        section is missing or malformed, the requested atom does not have
        exactly one ``type vdos`` section, or unsupported DYNINFO fields
        prevent safe replacement.
    """
    # Reject malformed VDOS before changing any part of the NCMAT document.
    grid, vals = _validate_vdos_input(energy_ev, density)
    ensure_custom_liquids(text)

    # Keep all non-target sections exactly as supplied, including plugin data.
    header, sections = _split_ncmat(text)
    found = 0
    result = []
    for tag, body in sections:
        if tag != "DYNINFO":
            result.append(body)
            continue
        # Select a DYNINFO section by species, not by its position in the file.
        match = re.search(r"(?m)^\s*element\s+(\S+)\s*$", body)
        if match is None or match.group(1) != atom:
            result.append(body)
            continue
        found += 1
        if not re.search(r"(?m)^\s*type\s+vdos\s*$", body):
            raise ValueError(f"Target atom {atom} must have @DYNINFO type vdos.")
        result.append(_render_vdos_dyninfo(body, grid, vals))

    # Multiple matches risk modifying the wrong representation or atom site.
    if found != 1:
        raise ValueError(f"Expected exactly one @DYNINFO VDOS for {atom}; found {found}.")
    rebuilt = header + "".join(result)

    # Revalidate after replacement to ensure plugin settings survived intact.
    ensure_custom_liquids(rebuilt)
    return rebuilt


def load_liquid_ncmat(path: str | Path) -> str:
    """Load NCMAT text and validate its required liquid plugin extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the NCMAT text file, read with UTF-8 encoding.

    Returns
    -------
    str
        Original NCMAT text, with no rewriting or normalization.

    Raises
    ------
    OSError
        If the file cannot be opened or read.
    UnicodeError
        If the file cannot be decoded as UTF-8.
    ValueError
        If NCMAT framing or ``@CUSTOM_LIQUIDS`` entries are invalid.
    """
    # Enforce the plugin-specific extension before downstream NCrystal loading.
    text = Path(path).read_text(encoding="utf-8")
    ensure_custom_liquids(text)
    return text
