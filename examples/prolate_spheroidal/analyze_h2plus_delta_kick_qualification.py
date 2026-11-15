"""Analyze the production qualification matrix for H2+ delta kicks.

The input directory must contain the ``reference``, ``timestep``,
``cap_onset``, ``cap_strength``, and ``large_box`` archives produced by
``h2plus_delta_kick_spectrum.py``.  The report collects the Step-11
acceptance metrics in a machine-readable JSON file and exits unsuccessfully
if a required check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grid_tdse.prolate_spheroidal import (
    windowed_positive_frequency_spectrum,
)


REFERENCE_STRENGTH = "K1.000em03"
WEAK_STRENGTH = "K5.000em04"
REQUIRED_CASES = (
    "reference",
    "timestep",
    "cap_onset",
    "cap_strength",
    "large_box",
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("output/step11_qualification"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON destination (default: ROOT/qualification_report.json)",
    )
    return parser.parse_args()


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["metadata"] = json.loads(str(arrays.pop("metadata_json")))
    return arrays


def relative_l2(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    denominator = np.linalg.norm(right)
    if denominator == 0.0:
        return float(np.linalg.norm(left - right))
    return float(np.linalg.norm(left - right) / denominator)


def local_peak_frequency(frequencies, transform, target, radius):
    frequencies = np.asarray(frequencies)
    power = np.abs(transform) ** 2
    maxima = np.flatnonzero(
        (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    ) + 1
    candidates = maxima[abs(frequencies[maxima] - target) <= radius]
    if not candidates.size:
        raise RuntimeError(f"no local peak found near {target:.12f} E_h")
    index = candidates[np.argmin(abs(frequencies[candidates] - target))]
    return float(frequencies[index]), float(power[index])


def spectrum_metrics(frequencies, transform, expected):
    frequencies = np.asarray(frequencies)
    spacing = float(frequencies[1] - frequencies[0])
    peaks = {}
    for label, target in expected.items():
        frequency, power = local_peak_frequency(
            frequencies,
            transform,
            target,
            radius=max(3.0 * spacing, 0.006),
        )
        peaks[label] = {
            "expected": float(target),
            "frequency": frequency,
            "absolute_error": abs(frequency - target),
            "power": power,
        }
    return spacing, peaks


def restricted_spectrum(archive, final_time):
    times = archive["times"]
    selected = times <= final_time + 1.0e-12
    return windowed_positive_frequency_spectrum(
        times[selected],
        archive[f"response_{REFERENCE_STRENGTH}"][selected],
        window="hann",
    )


def compare_response(reference, candidate):
    candidate_times = candidate["times"]
    reference_times = reference["times"]
    reference_response = reference[f"response_{REFERENCE_STRENGTH}"]
    interpolated = np.interp(
        candidate_times,
        reference_times,
        reference_response,
    )
    return relative_l2(
        candidate[f"response_{REFERENCE_STRENGTH}"], interpolated
    )


def final_no_kick_norm_error(archive):
    return abs(float(archive["no_kick_norm"][-1]) - 1.0)


def main():
    arguments = parse_arguments()
    destination = arguments.output
    if destination is None:
        destination = arguments.root / "qualification_report.json"

    cases = {
        name: load_archive(arguments.root / name / "response.npz")
        for name in REQUIRED_CASES
    }
    reference = cases["reference"]
    metadata = reference["metadata"]
    allowed = metadata["expected_allowed_excitations"]
    forbidden = metadata["expected_forbidden_excitations"]

    full_spacing, full_peaks = spectrum_metrics(
        reference[f"omega_{REFERENCE_STRENGTH}"],
        reference[f"transform_{REFERENCE_STRENGTH}"],
        allowed,
    )
    late_spacing, late_peaks = spectrum_metrics(
        reference[f"late_omega_{REFERENCE_STRENGTH}"],
        reference[f"late_transform_{REFERENCE_STRENGTH}"],
        allowed,
    )
    full_late_shifts = {
        label: abs(
            full_peaks[label]["frequency"]
            - late_peaks[label]["frequency"]
        )
        for label in allowed
    }

    full_frequencies = reference[f"omega_{REFERENCE_STRENGTH}"]
    full_power = abs(reference[f"transform_{REFERENCE_STRENGTH}"]) ** 2
    forbidden_power = {}
    weakest_allowed_power = min(
        result["power"] for result in full_peaks.values()
    )
    for label, energy in forbidden.items():
        index = int(np.argmin(abs(full_frequencies - energy)))
        forbidden_power[label] = {
            "expected": float(energy),
            "sampled_frequency": float(full_frequencies[index]),
            "power": float(full_power[index]),
            "ratio_to_weakest_allowed_peak": float(
                full_power[index] / weakest_allowed_power
            ),
        }

    reference_times = reference["times"]
    cap_rate = reference[
        f"positive_cap_norm_loss_rate_{REFERENCE_STRENGTH}"
    ]
    cap_probability = reference[
        f"positive_absorber_region_probability_{REFERENCE_STRENGTH}"
    ]
    final_quarter = reference_times >= 0.75 * reference_times[-1]
    cap_stabilization = {
        "late_mean_rate_to_peak": float(
            np.mean(cap_rate[final_quarter]) / np.max(cap_rate)
        ),
        "late_mean_region_probability_to_peak": float(
            np.mean(cap_probability[final_quarter])
            / np.max(cap_probability)
        ),
    }

    baseline_1000 = restricted_spectrum(reference, 1000.0)
    baseline_spacing, baseline_peaks = spectrum_metrics(
        baseline_1000.angular_frequencies,
        baseline_1000.transform,
        dict(list(allowed.items())[:2]),
    )
    comparisons = {}
    for name in REQUIRED_CASES[1:]:
        candidate = cases[name]
        candidate_spacing, candidate_peaks = spectrum_metrics(
            candidate[f"omega_{REFERENCE_STRENGTH}"],
            candidate[f"transform_{REFERENCE_STRENGTH}"],
            dict(list(allowed.items())[:2]),
        )
        comparisons[name] = {
            "response_relative_l2": compare_response(reference, candidate),
            "no_kick_final_norm_error": final_no_kick_norm_error(candidate),
            "frequency_spacing": candidate_spacing,
            "peak_shifts_from_reference_T1000": {
                label: abs(
                    candidate_peaks[label]["frequency"]
                    - baseline_peaks[label]["frequency"]
                )
                for label in baseline_peaks
            },
            "peaks": candidate_peaks,
        }

    antisymmetry = {
        label: float(
            metadata["responses"][label][
                "kick_sign_antisymmetry_relative_l2"
            ]
        )
        for label in (REFERENCE_STRENGTH, WEAK_STRENGTH)
    }
    linearity = float(
        metadata["linearity_relative_l2_to_reference"][WEAK_STRENGTH]
    )
    no_kick_error = final_no_kick_norm_error(reference)

    checks = {
        "kick_sign_antisymmetry": max(antisymmetry.values()) < 1.0e-6,
        "kick_strength_linearity": linearity < 1.0e-4,
        "no_kick_norm": no_kick_error < 1.0e-10,
        "allowed_full_peak_positions": max(
            result["absolute_error"] for result in full_peaks.values()
        )
        <= full_spacing,
        "allowed_late_peak_positions": max(
            result["absolute_error"] for result in late_peaks.values()
        )
        <= late_spacing,
        "full_late_peak_agreement": max(full_late_shifts.values())
        <= max(full_spacing, late_spacing),
        "forbidden_peak_absent": max(
            result["ratio_to_weakest_allowed_peak"]
            for result in forbidden_power.values()
        )
        < 1.0e-3,
        "cap_loss_stabilized": cap_stabilization[
            "late_mean_rate_to_peak"
        ]
        < 1.0e-3,
        "cap_region_stabilized": cap_stabilization[
            "late_mean_region_probability_to_peak"
        ]
        < 1.0e-3,
        "time_step_stability": comparisons["timestep"][
            "response_relative_l2"
        ]
        < 2.0e-4,
        "cap_onset_stability": comparisons["cap_onset"][
            "response_relative_l2"
        ]
        < 1.0e-3,
        "cap_strength_stability": comparisons["cap_strength"][
            "response_relative_l2"
        ]
        < 1.0e-3,
        "box_stability": comparisons["large_box"][
            "response_relative_l2"
        ]
        < 1.0e-3,
        "control_no_kick_norms": max(
            result["no_kick_final_norm_error"]
            for result in comparisons.values()
        )
        < 1.0e-10,
        "control_peak_stability": max(
            shift
            for result in comparisons.values()
            for shift in result["peak_shifts_from_reference_T1000"].values()
        )
        <= max(
            baseline_spacing,
            *(result["frequency_spacing"] for result in comparisons.values()),
        ),
    }

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "kick_sign_antisymmetry_relative_l2": antisymmetry,
        "kick_strength_linearity_relative_l2": linearity,
        "reference_no_kick_final_norm_error": no_kick_error,
        "reference_full_frequency_spacing": full_spacing,
        "reference_late_frequency_spacing": late_spacing,
        "reference_full_peaks": full_peaks,
        "reference_late_peaks": late_peaks,
        "reference_full_late_peak_shifts": full_late_shifts,
        "forbidden_frequency_power": forbidden_power,
        "cap_stabilization": cap_stabilization,
        "control_comparisons": comparisons,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Step 11 production qualification: {'PASS' if report['passed'] else 'FAIL'}")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nReport: {destination}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
