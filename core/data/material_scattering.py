"""Backend-independent physical model for sampled scattering materials.

The model describes physical contributions, not LEAPR or NCMAT cards.  A
backend may implement the stated model only when it supports its semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from common.validation import require_finite_nonnegative, require_finite_positive
from core.data.input_pdos import PhononSpectra


def _finite(value: float, label: str, *, positive: bool = False) -> None:
    """Check a finite physical scalar.

    Parameters
    ----------
    value : float
        Value to validate.
    label : str
        Name shown when validation fails.
    positive : bool, optional
        Require strictly positive values instead of nonnegative values.
    """
    if positive:
        require_finite_positive(value, name=label)
    else:
        require_finite_nonnegative(value, name=label)


@dataclass(frozen=True)
class ScatteringSpecies:
    """A species and its multiplicity in a physical scattering material.

    Parameters
    ----------
    label : str
        Stable, user-provided species identifier (e.g. ``H`` or ``O``).
    multiplicity : int
        Number of equivalent atoms per model unit.
    mass_ratio : float
        Atomic mass divided by neutron mass (dimensionless).
    """

    label: str
    multiplicity: int
    mass_ratio: float

    def __post_init__(self) -> None:
        """Reject nonphysical or ambiguous species data."""
        if not self.label.strip() or type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("Species label and positive integer multiplicity are required.")
        _finite(self.mass_ratio, "Species mass ratio", positive=True)


@dataclass(frozen=True)
class DiscreteOscillator:
    """One identifiable vibrational contribution.

    Parameters
    ----------
    label : str
        Stable oscillator identity, independent of array order.
    energy_ev : float
        Oscillator energy in electronvolts.
    weight : float
        Fraction of the dynamical contribution.
    """

    label: str
    energy_ev: float
    weight: float

    def __post_init__(self) -> None:
        """Validate oscillator parameters."""
        if not self.label.strip():
            raise ValueError("An oscillator requires a stable label.")
        _finite(self.energy_ev, "Oscillator energy", positive=True)
        _finite(self.weight, "Oscillator weight")


@dataclass(frozen=True)
class SpeciesDynamics:
    """Independent-motion model assigned to a principal scattering species.

    The continuous spectrum is *always* normalized independently of its
    contribution weight. Translation and vibrational scattering laws are
    composed by the numerical backend; their kernels are not simply added.

    Parameters
    ----------
    continuous : PhononSpectra
        Unit-integral continuous spectrum in eV.
    continuous_weight : float
        Weight of the continuous vibrational contribution.
    translation_weight : float
        Weight of translational motion.
    translation_model : str
        ``none``, ``free_gas`` or the physical ``egelstaff_schofield``
        effective-width translational model.
    translation_parameter : float or None
        Dimensionless Egelstaff--Schofield diffusion parameter ``c`` for
        that physical model; not a generic SI diffusion coefficient.
    oscillators : tuple of DiscreteOscillator
        Optional separate intramolecular vibrational modes.
    """

    # Continuous spectrum and component weights.
    continuous: PhononSpectra
    continuous_weight: float = 1.0
    translation_weight: float = 0.0

    # Translation model and fixed oscillator modes.
    translation_model: str = "none"
    oscillators: tuple[DiscreteOscillator, ...] = ()
    translation_parameter: float | None = None

    def __post_init__(self) -> None:
        """Enforce normalized independent physical contributions."""
        # Keep the continuum shape normalized independently of its physical weight.
        if not isinstance(self.continuous, PhononSpectra):
            raise TypeError("continuous must be a PhononSpectra instance.")
        if not math.isclose(self.continuous.integral, 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Continuous spectrum must be normalized to unit area.")

        # Validate the individual weights before testing their sum.
        _finite(self.continuous_weight, "Continuous weight")
        _finite(self.translation_weight, "Translation weight")
        self._validate_translation()
        self._validate_component_weights()

    def _validate_translation(self) -> None:
        """Check translation model, parameter, and weight consistency.

        Raises
        ------
        ValueError
            If the translation model or its parameters are inconsistent.
        """
        # Only the Egelstaff–Schofield model carries an explicit diffusion parameter.
        if self.translation_model not in ("none", "free_gas", "egelstaff_schofield"):
            raise ValueError("Unsupported translation model.")
        if self.translation_model == "egelstaff_schofield":
            if self.translation_parameter is None:
                raise ValueError("Egelstaff--Schofield translation requires parameter c.")
            _finite(self.translation_parameter, "Egelstaff--Schofield c", positive=True)
        elif self.translation_parameter is not None:
            raise ValueError("A diffusion parameter requires Egelstaff--Schofield translation.")
        # A zero translation weight goes with the no-translation model.
        if (self.translation_model == "none") != (self.translation_weight == 0):
            raise ValueError("Translation model and translation weight are inconsistent.")

    def _validate_component_weights(self) -> None:
        """Validate oscillator identities and the combined component weight.

        Raises
        ------
        ValueError
            If oscillator labels repeat or component weights do not sum to unity within tolerance.
        """
        # Keep oscillator identities unambiguous when matching individual modes.
        names = [osc.label for osc in self.oscillators]
        if len(names) != len(set(names)):
            raise ValueError("Oscillator labels must be unique.")
        total = self.continuous_weight + self.translation_weight + sum(
            osc.weight for osc in self.oscillators
        )
        # Preserve CAB/JEFF tabulated coefficients rounded to 5-6 decimals:
        # e.g. 0.005754 + 0.52425 + 0.15667 + 0.31333 = 1.000004.
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f"Dynamical weights must sum to unity; got {total:.9g}.")

    def with_spectrum(self, spectrum: PhononSpectra) -> "SpeciesDynamics":
        """Replace only the continuous spectrum, retaining fixed contributions.

        Parameters
        ----------
        spectrum : PhononSpectra
            New normalized stochastic realization.

        Returns
        -------
        SpeciesDynamics
            Validated independent realization.
        """
        return replace(self, continuous=spectrum)


@dataclass(frozen=True)
class MaterialScatteringModel:
    """Backend-independent material with one selected principal dynamical law.

    Secondary species describe material composition without assuming a
    dynamical representation they do not have. Additional species-specific
    models can be introduced when supported by verified backend physics.

    Parameters
    ----------
    identifier : str
        Material label, not an ENDF MAT identifier.
    state : str
        ``solid`` or ``liquid`` metadata; contributions determine physics.
    temperature_k : float
        Temperature in kelvin.
    principal : ScatteringSpecies
        Scatterer to which the specified dynamical law applies.
    dynamics : SpeciesDynamics
        Principal scatterer's independent-motion model.
    secondary : tuple of ScatteringSpecies
        Other material species represented by the input material model.
    """

    # Material identity and conditions.
    identifier: str
    state: str
    temperature_k: float

    # Scattering species and dynamics.
    principal: ScatteringSpecies
    dynamics: SpeciesDynamics
    secondary: tuple[ScatteringSpecies, ...] = ()

    def __post_init__(self) -> None:
        """Validate composition and thermodynamic state."""
        if not self.identifier.strip() or self.state not in ("solid", "liquid"):
            raise ValueError("A material identifier and state (solid/liquid) are required.")
        _finite(self.temperature_k, "Temperature", positive=True)

        # Species labels must identify a single component across both scatterer
        # groups.
        names = [self.principal.label] + [item.label for item in self.secondary]
        if len(names) != len(set(names)):
            raise ValueError("Material scattering species must have unique labels.")

    def with_dynamics(self, dynamics: SpeciesDynamics) -> "MaterialScatteringModel":
        """Return a new material with revised validated principal dynamics.

        Parameters
        ----------
        dynamics : SpeciesDynamics
            Replacement independent-motion model.

        Returns
        -------
        MaterialScatteringModel
            Immutable material realization preserving material identity.
        """
        return replace(self, dynamics=dynamics)

    def with_spectrum(self, spectrum: PhononSpectra) -> "MaterialScatteringModel":
        """Create an immutable model realization for one stochastic spectrum.

        Parameters
        ----------
        spectrum : PhononSpectra
            Sampled unit-normalized spectrum.

        Returns
        -------
        MaterialScatteringModel
            New valid model with unchanged composition, weights and modes.
        """
        return replace(self, dynamics=self.dynamics.with_spectrum(spectrum))
