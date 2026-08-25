"""Deterministic evaluation for a trained CityLearn DRL policy.

This module evaluates one trained policy on one configured building using the
revised NormCCEReward with shared global normalization references:

    avoidable_export_reference
    net_consumption_reference
    cost_reference
    emission_reference

The implementation validates:

1. Reward configuration and reference values.
2. Model/environment observation and action-space compatibility.
3. Transition timing and physical identities.
4. Reward normalization and scalar-reward reconstruction.
5. SOC fairness and normalization-reference exceedances.
6. Complete time-series, KPI, and configuration exports.

Compatibility target:
    Python 3.7
    CityLearn 1.4.4
    NumPy 1.21.6
    pandas 1.3.5
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from citylearn.cost_function import CostFunction
from environment.citylearn_env import SocFeasibleCityLearnEnv


EPSILON = 1e-8
REFERENCE_TOLERANCE = 1e-6


# =============================================================================
# BASIC HELPERS
# =============================================================================


def get_scalar(value):
    """Return the first numerical scalar from a nested scalar-like value."""

    array = np.asarray(value, dtype=float).reshape(-1)

    if array.size == 0:
        raise ValueError("Cannot extract a scalar from an empty value.")

    if not np.all(np.isfinite(array)):
        raise ValueError("Scalar-like value contains NaN or infinity.")

    return float(array[0])


def safe_ratio(numerator, denominator):
    """Return a finite ratio using a small positive denominator offset."""

    numerator = float(numerator)
    denominator = float(denominator)

    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("Ratio inputs must be finite.")

    return numerator / (denominator + EPSILON)


def safe_json_dump(data, filepath):
    """Write JSON using standard Python numerical types."""

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    with open(filepath, "w", encoding="utf-8") as stream:
        json.dump(convert(data), stream, indent=4)


# =============================================================================
# REWARD CONFIGURATION
# =============================================================================


def get_reward_configuration(settings, building_id):
    """Return the validated global NormCCEReward configuration.

    The building identifier is used only for logging. All buildings receive
    the same reward weights and the same global normalization references.
    """

    building_name = "Building_{}".format(building_id)

    if "reward_functions" not in settings:
        raise KeyError("settings.json does not contain 'reward_functions'.")

    if "NormCCEReward" not in settings["reward_functions"]:
        raise KeyError(
            "settings.json does not contain reward_functions/NormCCEReward."
        )

    reward_settings = settings["reward_functions"]["NormCCEReward"]

    if "type" not in reward_settings:
        raise KeyError("NormCCEReward settings are missing 'type'.")

    if "attributes" not in reward_settings:
        raise KeyError("NormCCEReward settings are missing 'attributes'.")

    reward_attributes = reward_settings["attributes"].copy()

    if "building_upper_bounds" in reward_settings:
        raise KeyError(
            "The obsolete 'building_upper_bounds' block is still present. "
            "Evaluation must use one shared global attributes block."
        )

    if "energy_reference" in reward_attributes:
        raise KeyError(
            "The obsolete 'energy_reference' field is still present. "
            "Replace it with 'avoidable_export_reference'."
        )

    required_attributes = [
        "grid_weight",
        "cost_weight",
        "emission_weight",
        "avoidable_export_reference",
        "net_consumption_reference",
        "cost_reference",
        "emission_reference",
        "soc_min",
        "soc_max",
        "export_tariff",
        "terminal_soc_weight",
        "epsilon",
    ]

    missing_attributes = [
        name for name in required_attributes if name not in reward_attributes
    ]

    if missing_attributes:
        raise KeyError(
            "Missing NormCCEReward attributes: {}".format(missing_attributes)
        )

    numerical_attributes = list(required_attributes)

    for name in numerical_attributes:
        try:
            value = float(reward_attributes[name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "NormCCEReward attribute {!r} must be numeric. "
                "Received {!r}.".format(name, reward_attributes[name])
            ) from error

        if not np.isfinite(value):
            raise ValueError(
                "NormCCEReward attribute {!r} must be finite.".format(name)
            )

        reward_attributes[name] = value

    reward_attributes.setdefault("validation_tolerance", 1e-6)
    reward_attributes.setdefault("store_diagnostic_history", False)

    reward_attributes["validation_tolerance"] = float(
        reward_attributes["validation_tolerance"]
    )
    reward_attributes["store_diagnostic_history"] = bool(
        reward_attributes["store_diagnostic_history"]
    )

    weights = np.asarray(
        [
            reward_attributes["grid_weight"],
            reward_attributes["cost_weight"],
            reward_attributes["emission_weight"],
        ],
        dtype=float,
    )

    if np.any(weights < 0.0):
        raise ValueError("NormCCEReward weights must be non-negative.")

    if not np.isclose(weights.sum(), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "NormCCEReward weights must sum to 1.0. "
            "Received sum: {:.10f}.".format(weights.sum())
        )

    references = np.asarray(
        [
            reward_attributes["avoidable_export_reference"],
            reward_attributes["net_consumption_reference"],
            reward_attributes["cost_reference"],
            reward_attributes["emission_reference"],
        ],
        dtype=float,
    )

    if np.any(references <= 0.0):
        raise ValueError(
            "All NormCCEReward normalization references must be positive."
        )

    if not (
        0.0
        <= reward_attributes["soc_min"]
        < reward_attributes["soc_max"]
        <= 1.0
    ):
        raise ValueError(
            "SOC limits must satisfy 0 <= soc_min < soc_max <= 1."
        )

    if reward_attributes["export_tariff"] < 0.0:
        raise ValueError("export_tariff must be non-negative.")

    if reward_attributes["terminal_soc_weight"] < 0.0:
        raise ValueError("terminal_soc_weight must be non-negative.")

    if reward_attributes["epsilon"] <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    if reward_attributes["validation_tolerance"] <= 0.0:
        raise ValueError("validation_tolerance must be greater than zero.")

    print(
        "Using global NormCCEReward evaluation parameters for {}:".format(
            building_name
        )
    )
    print("  Grid weight:", reward_attributes["grid_weight"])
    print("  Cost weight:", reward_attributes["cost_weight"])
    print("  Emission weight:", reward_attributes["emission_weight"])
    print(
        "  Avoidable-export reference:",
        reward_attributes["avoidable_export_reference"],
    )
    print(
        "  Net-consumption reference:",
        reward_attributes["net_consumption_reference"],
    )
    print("  Cost reference:", reward_attributes["cost_reference"])
    print("  Emission reference:", reward_attributes["emission_reference"])
    print(
        "  SOC range: [{}, {}]".format(
            reward_attributes["soc_min"],
            reward_attributes["soc_max"],
        )
    )
    print("  Export tariff:", reward_attributes["export_tariff"])
    print(
        "  Terminal SOC weight:",
        reward_attributes["terminal_soc_weight"],
    )

    return {
        "type": reward_settings["type"],
        "attributes": reward_attributes,
    }


def load_eval_schema_from_settings():
    """Construct a one-building evaluation schema from settings.json."""

    settings_path = os.path.join(os.path.dirname(__file__), "settings.json")

    if not os.path.isfile(settings_path):
        raise FileNotFoundError(
            "Could not find evaluation settings: {}".format(settings_path)
        )

    with open(settings_path, "r", encoding="utf-8") as stream:
        settings = json.load(stream)

    if "evaluation" not in settings:
        raise KeyError("settings.json does not contain 'evaluation'.")

    evaluation_settings = settings["evaluation"]

    required_evaluation_fields = [
        "building_id",
        "schema_filename",
        "start_time_step",
        "end_time_step",
    ]
    missing_evaluation_fields = [
        name
        for name in required_evaluation_fields
        if name not in evaluation_settings
    ]

    if missing_evaluation_fields:
        raise KeyError(
            "Missing evaluation settings: {}".format(
                missing_evaluation_fields
            )
        )

    evaluation_building = int(evaluation_settings["building_id"])
    evaluation_start = int(evaluation_settings["start_time_step"])
    evaluation_end = int(evaluation_settings["end_time_step"])

    if evaluation_start < 0 or evaluation_end < evaluation_start:
        raise ValueError(
            "Invalid evaluation interval [{}, {}].".format(
                evaluation_start,
                evaluation_end,
            )
        )

    schema_filename = evaluation_settings["schema_filename"]
    schema_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "misc",
            "exp_schema",
            schema_filename,
        )
    )

    if not os.path.isfile(schema_path):
        raise FileNotFoundError(
            "Could not find evaluation schema: {}".format(schema_path)
        )

    with open(schema_path, "r", encoding="utf-8") as stream:
        schema = json.load(stream)

    if "buildings" not in schema:
        raise KeyError("Evaluation schema does not contain 'buildings'.")

    schema["simulation_start_time_step"] = evaluation_start
    schema["simulation_end_time_step"] = evaluation_end
    schema["episodes"] = 1
    schema["central_agent"] = settings["central_agent"]

    matched_building = False

    for building_name in schema["buildings"]:
        building_id = int(building_name.split("_")[-1])
        include_building = building_id == evaluation_building
        schema["buildings"][building_name]["include"] = include_building
        matched_building = matched_building or include_building

    if not matched_building:
        raise KeyError(
            "Evaluation building {} is absent from the schema.".format(
                evaluation_building
            )
        )

    schema["reward_function"] = get_reward_configuration(
        settings,
        evaluation_building,
    )

    # The trained model is loaded explicitly, so CityLearn should not create a
    # new evaluation agent from the schema.
    schema.pop("agent", None)

    return schema, settings


# =============================================================================
# MODEL/ENVIRONMENT COMPATIBILITY
# =============================================================================


def validate_model_environment_compatibility(model, environment):
    """Verify observation ordering and action-space compatibility."""

    required_model_attributes = [
        "observation_names",
        "encoders",
        "action_space",
    ]
    missing_model_attributes = [
        name for name in required_model_attributes if not hasattr(model, name)
    ]

    if missing_model_attributes:
        raise AttributeError(
            "The loaded model is missing attributes: {}".format(
                missing_model_attributes
            )
        )

    if len(model.observation_names) != len(environment.observation_names):
        raise ValueError(
            "The model and environment have different agent counts."
        )

    for index, (model_names, environment_names) in enumerate(
        zip(model.observation_names, environment.observation_names)
    ):
        print("\nAgent {} model observations:".format(index))
        print(model_names)
        print("Model observation count:", len(model_names))
        print("\nAgent {} environment observations:".format(index))
        print(environment_names)
        print("Environment observation count:", len(environment_names))

        if list(model_names) != list(environment_names):
            raise ValueError(
                "Observation mismatch for agent {}.\n"
                "Model observations: {}\n"
                "Environment observations: {}\n"
                "Training and evaluation must use identical observation "
                "names and ordering.".format(
                    index,
                    model_names,
                    environment_names,
                )
            )

        encoder_count = len(model.encoders[index])

        if encoder_count != len(environment_names):
            raise ValueError(
                "Encoder mismatch for agent {}: {} encoders for {} "
                "observations.".format(
                    index,
                    encoder_count,
                    len(environment_names),
                )
            )

    if len(model.action_space) != len(environment.action_space):
        raise ValueError(
            "The model and environment have different action-space counts."
        )

    for index, (model_space, environment_space) in enumerate(
        zip(model.action_space, environment.action_space)
    ):
        if model_space.shape != environment_space.shape:
            raise ValueError(
                "Action-space mismatch for agent {}: model={}, "
                "environment={}.".format(
                    index,
                    model_space.shape,
                    environment_space.shape,
                )
            )

    print("\nModel-environment compatibility check passed.")


def validate_reward_instance(reward_function, expected_attributes):
    """Verify that the instantiated reward matches settings.json."""

    if reward_function.__class__.__name__ != "NormCCEReward":
        raise TypeError(
            "The evaluation environment must use NormCCEReward. "
            "Received {}.".format(reward_function.__class__.__name__)
        )

    attributes_to_compare = [
        "grid_weight",
        "cost_weight",
        "emission_weight",
        "avoidable_export_reference",
        "net_consumption_reference",
        "cost_reference",
        "emission_reference",
        "soc_min",
        "soc_max",
        "export_tariff",
        "terminal_soc_weight",
        "epsilon",
    ]

    for name in attributes_to_compare:
        if not hasattr(reward_function, name):
            raise AttributeError(
                "Instantiated NormCCEReward is missing {!r}.".format(name)
            )

        actual = float(getattr(reward_function, name))
        expected = float(expected_attributes[name])

        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
            raise ValueError(
                "Reward configuration mismatch for {!r}: "
                "expected {:.10f}, received {:.10f}.".format(
                    name,
                    expected,
                    actual,
                )
            )

    print("Global NormCCEReward references validated successfully.")


# =============================================================================
# TRANSITION AND REWARD VALIDATION
# =============================================================================


def validate_transition_timing(context):
    """Verify the action/reward time-step alignment."""

    required_fields = ["action_time_step", "reward_time_step"]
    missing_fields = [name for name in required_fields if name not in context]

    if missing_fields:
        raise KeyError(
            "Missing transition-context time fields: {}".format(
                missing_fields
            )
        )

    action_time_step = int(context["action_time_step"])
    reward_time_step = int(context["reward_time_step"])

    if reward_time_step != action_time_step + 1:
        raise ValueError(
            "Unexpected CityLearn transition timing: "
            "action_time_step={}, reward_time_step={}. "
            "Expected reward_time_step = action_time_step + 1.".format(
                action_time_step,
                reward_time_step,
            )
        )

    return action_time_step, reward_time_step


def validate_reward_diagnostics(diagnostics, reward_value):
    """Validate physical identities and the reconstructed scalar reward."""

    required_numerical_fields = [
        "pre_action_soc",
        "post_action_soc",
        "initial_episode_soc",
        "load",
        "pv_generation",
        "base_net_demand",
        "maximum_charge_energy",
        "maximum_discharge_energy",
        "feasible_lower_grid_limit",
        "feasible_upper_grid_limit",
        "projected_grid_target",
        "battery_consumption",
        "net_consumption",
        "grid_import",
        "grid_export",
        "unavoidable_export",
        "avoidable_export",
        "raw_grid_penalty",
        "raw_cost",
        "raw_emission",
        "avoidable_export_reference",
        "net_consumption_reference",
        "cost_reference",
        "emission_reference",
        "normalized_grid",
        "normalized_grid_import",
        "normalized_cost",
        "normalized_emission",
        "weighted_grid",
        "weighted_cost",
        "weighted_emission",
        "feasible_grid_span",
        "avoidable_export_fraction_of_flexibility",
        "terminal_soc_penalty",
        "reward",
    ]

    required_boolean_fields = [
        "avoidable_export_exceeds_flexibility",
        "grid_reference_exceeded",
        "net_consumption_reference_exceeded",
        "cost_reference_exceeded",
        "emission_reference_exceeded",
        "terminal",
    ]

    missing_fields = [
        name
        for name in required_numerical_fields + required_boolean_fields
        if name not in diagnostics
    ]

    if missing_fields:
        raise KeyError(
            "Missing NormCCEReward diagnostics: {}".format(missing_fields)
        )

    numerical_values = np.asarray(
        [diagnostics[name] for name in required_numerical_fields],
        dtype=float,
    )

    if not np.all(np.isfinite(numerical_values)):
        raise ValueError(
            "NormCCEReward diagnostics contain NaN or infinite values."
        )

    expected_net = (
        diagnostics["base_net_demand"]
        + diagnostics["battery_consumption"]
    )

    if not np.isclose(
        diagnostics["net_consumption"],
        expected_net,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "Net-consumption identity failed during evaluation."
        )

    expected_grid_import = max(diagnostics["net_consumption"], 0.0)
    expected_grid_export = max(-diagnostics["net_consumption"], 0.0)
    expected_unavoidable_export = max(
        -diagnostics["projected_grid_target"],
        0.0,
    )
    expected_avoidable_export = max(
        expected_grid_export - expected_unavoidable_export,
        0.0,
    )

    checks = {
        "grid import": (
            diagnostics["grid_import"],
            expected_grid_import,
        ),
        "grid export": (
            diagnostics["grid_export"],
            expected_grid_export,
        ),
        "unavoidable export": (
            diagnostics["unavoidable_export"],
            expected_unavoidable_export,
        ),
        "avoidable export": (
            diagnostics["avoidable_export"],
            expected_avoidable_export,
        ),
    }

    for name, (actual, expected) in checks.items():
        if not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
            raise ValueError(
                "{} consistency check failed during evaluation. "
                "Actual={:.10f}, expected={:.10f}.".format(
                    name.title(),
                    actual,
                    expected,
                )
            )

    normalization_checks = {
        "normalized grid": (
            diagnostics["normalized_grid"],
            diagnostics["raw_grid_penalty"]
            / (diagnostics["avoidable_export_reference"] + EPSILON),
        ),
        "normalized grid import": (
            diagnostics["normalized_grid_import"],
            diagnostics["grid_import"]
            / (diagnostics["net_consumption_reference"] + EPSILON),
        ),
        "normalized cost": (
            diagnostics["normalized_cost"],
            diagnostics["raw_cost"]
            / (diagnostics["cost_reference"] + EPSILON),
        ),
        "normalized emission": (
            diagnostics["normalized_emission"],
            diagnostics["raw_emission"]
            / (diagnostics["emission_reference"] + EPSILON),
        ),
    }

    for name, (actual, expected) in normalization_checks.items():
        if not np.isclose(actual, expected, rtol=1e-8, atol=1e-8):
            raise ValueError(
                "{} identity failed during evaluation. "
                "Actual={:.10f}, expected={:.10f}.".format(
                    name.title(),
                    actual,
                    expected,
                )
            )

    expected_reward = -(
        diagnostics["weighted_grid"]
        + diagnostics["weighted_cost"]
        + diagnostics["weighted_emission"]
    ) + diagnostics["terminal_soc_penalty"]

    if not np.isclose(
        reward_value,
        expected_reward,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError(
            "Reward identity failed during evaluation. "
            "Environment reward={:.10f}, expected reward={:.10f}.".format(
                reward_value,
                expected_reward,
            )
        )

    if not np.isclose(
        diagnostics["reward"],
        reward_value,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError(
            "Reward diagnostic does not match the environment reward."
        )


# =============================================================================
# KPI HELPERS
# =============================================================================


def calculate_reference_diagnostics(timeseries):
    """Summarize normalization-reference exceedances."""

    count_columns = {
        "grid_reference_exceedance_count": "grid_reference_exceeded",
        "net_consumption_reference_exceedance_count": (
            "net_consumption_reference_exceeded"
        ),
        "cost_reference_exceedance_count": "cost_reference_exceeded",
        "emission_reference_exceedance_count": (
            "emission_reference_exceeded"
        ),
        "avoidable_export_flexibility_exceedance_count": (
            "avoidable_export_exceeds_flexibility"
        ),
    }

    result = {
        output_name: int(timeseries[column_name].astype(bool).sum())
        for output_name, column_name in count_columns.items()
    }

    result.update(
        {
            "maximum_normalized_grid": float(
                timeseries["normalized_grid"].max()
            ),
            "maximum_normalized_grid_import": float(
                timeseries["normalized_grid_import"].max()
            ),
            "maximum_normalized_cost": float(
                timeseries["normalized_cost"].max()
            ),
            "maximum_normalized_emission": float(
                timeseries["normalized_emission"].max()
            ),
        }
    )

    return result


def print_reference_diagnostics(reference_diagnostics):
    """Print a concise normalization-reference audit."""

    exceedance_keys = [
        "grid_reference_exceedance_count",
        "net_consumption_reference_exceedance_count",
        "cost_reference_exceedance_count",
        "emission_reference_exceedance_count",
    ]

    total_exceedances = sum(
        reference_diagnostics[name] for name in exceedance_keys
    )

    if total_exceedances == 0:
        print(
            "\nAll normalized reward quantities remained within their "
            "global references."
        )
    else:
        print(
            "\nWARNING: One or more global normalization references were "
            "exceeded during evaluation."
        )
        print(
            "  Avoidable-export exceedances:",
            reference_diagnostics["grid_reference_exceedance_count"],
        )
        print(
            "  Net-consumption exceedances:",
            reference_diagnostics[
                "net_consumption_reference_exceedance_count"
            ],
        )
        print(
            "  Cost exceedances:",
            reference_diagnostics["cost_reference_exceedance_count"],
        )
        print(
            "  Emission exceedances:",
            reference_diagnostics["emission_reference_exceedance_count"],
        )


# =============================================================================
# MAIN EVALUATION
# =============================================================================


def evaluate(
    env,
    model,
    simulation_id="evaluation",
    root_dir=".",
    deterministic=True,
    save=True,
):
    """Run deterministic one-building policy evaluation."""

    start_time = time.time()

    if not deterministic:
        raise ValueError(
            "Final policy evaluation must use deterministic=True."
        )

    if len(env.buildings) != 1:
        raise ValueError(
            "This evaluation implementation requires exactly one building."
        )

    observations = env.reset()
    building = env.buildings[0]
    reward_function = env.reward_function

    expected_attributes = env.schema["reward_function"]["attributes"]
    validate_reward_instance(reward_function, expected_attributes)

    if not hasattr(model, "get_post_exploration_actions"):
        raise AttributeError(
            "The loaded model does not provide "
            "get_post_exploration_actions()."
        )

    records = []

    while not env.done:
        environment_time_step_before_action = int(env.time_step)

        if isinstance(observations[0], list):
            model_observations = observations
        else:
            model_observations = [observations]

        # In this implementation, get_post_exploration_actions is the
        # deterministic policy path used after exploration has ended.
        action = model.get_post_exploration_actions(model_observations)

        next_observations, reward, done, _ = env.step(action)
        reward_value = get_scalar(reward)

        diagnostics = reward_function.latest_diagnostics.copy()

        if not diagnostics:
            raise RuntimeError(
                "NormCCEReward did not provide evaluation diagnostics."
            )

        validate_reward_diagnostics(diagnostics, reward_value)

        transition_context = reward_function.kwargs.get(
            "transition_context",
            {},
        )
        action_time_step, reward_time_step = validate_transition_timing(
            transition_context
        )

        if action_time_step != environment_time_step_before_action:
            raise ValueError(
                "The cached action time step does not match the environment "
                "time step observed before the action."
            )

        series_length = len(
            building.net_electricity_consumption_without_storage
        )

        if reward_time_step < 0 or reward_time_step >= series_length:
            raise IndexError(
                "reward_time_step={} lies outside the baseline series of "
                "length {}.".format(reward_time_step, series_length)
            )

        net_baseline = float(
            building.net_electricity_consumption_without_storage[
                reward_time_step
            ]
        )
        baseline_cost = float(
            building.net_electricity_consumption_without_storage_price[
                reward_time_step
            ]
        )
        baseline_emission = float(
            building.net_electricity_consumption_without_storage_emission[
                reward_time_step
            ]
        )

        baseline_grid_import = max(net_baseline, 0.0)
        positive_baseline_cost = max(baseline_cost, 0.0)
        positive_baseline_emission = max(baseline_emission, 0.0)

        positive_pv = diagnostics["pv_generation"]
        non_shiftable_load = diagnostics["load"]
        battery_consumption = diagnostics["battery_consumption"]

        pv_to_load = min(positive_pv, non_shiftable_load)
        pv_surplus = max(positive_pv - non_shiftable_load, 0.0)
        pv_to_battery = min(max(battery_consumption, 0.0), pv_surplus)
        pv_self_consumed = pv_to_load + pv_to_battery
        total_demand = non_shiftable_load + max(battery_consumption, 0.0)

        records.append(
            {
                "transition": len(records),
                "action_time_step": action_time_step,
                "reward_time_step": reward_time_step,
                "environment_time_step_after_action": int(env.time_step),
                "action": get_scalar(action),
                "reward": reward_value,
                "pre_action_soc": diagnostics["pre_action_soc"],
                "post_action_soc": diagnostics["post_action_soc"],
                "initial_episode_soc": diagnostics["initial_episode_soc"],
                "pre_action_soc_percent": (
                    diagnostics["pre_action_soc"] * 100.0
                ),
                "post_action_soc_percent": (
                    diagnostics["post_action_soc"] * 100.0
                ),
                "load": diagnostics["load"],
                "pv_generation": diagnostics["pv_generation"],
                "base_net_demand": diagnostics["base_net_demand"],
                "maximum_charge_energy": (
                    diagnostics["maximum_charge_energy"]
                ),
                "maximum_discharge_energy": (
                    diagnostics["maximum_discharge_energy"]
                ),
                "feasible_lower_grid_limit": (
                    diagnostics["feasible_lower_grid_limit"]
                ),
                "feasible_upper_grid_limit": (
                    diagnostics["feasible_upper_grid_limit"]
                ),
                "projected_grid_target": (
                    diagnostics["projected_grid_target"]
                ),
                "battery_consumption": diagnostics["battery_consumption"],
                "net_consumption": diagnostics["net_consumption"],
                "grid_import": diagnostics["grid_import"],
                "grid_export": diagnostics["grid_export"],
                "unavoidable_export": diagnostics["unavoidable_export"],
                "avoidable_export": diagnostics["avoidable_export"],
                "electricity_price": transition_context["buy_price"],
                "carbon_intensity": transition_context["carbon_intensity"],
                "raw_grid_penalty": diagnostics["raw_grid_penalty"],
                "raw_cost": diagnostics["raw_cost"],
                "raw_emission": diagnostics["raw_emission"],
                "avoidable_export_reference": (
                    diagnostics["avoidable_export_reference"]
                ),
                "net_consumption_reference": (
                    diagnostics["net_consumption_reference"]
                ),
                "cost_reference": diagnostics["cost_reference"],
                "emission_reference": diagnostics["emission_reference"],
                "normalized_grid": diagnostics["normalized_grid"],
                "normalized_grid_import": (
                    diagnostics["normalized_grid_import"]
                ),
                "normalized_cost": diagnostics["normalized_cost"],
                "normalized_emission": diagnostics["normalized_emission"],
                "weighted_grid": diagnostics["weighted_grid"],
                "weighted_cost": diagnostics["weighted_cost"],
                "weighted_emission": diagnostics["weighted_emission"],
                "feasible_grid_span": diagnostics["feasible_grid_span"],
                "avoidable_export_fraction_of_flexibility": diagnostics[
                    "avoidable_export_fraction_of_flexibility"
                ],
                "avoidable_export_exceeds_flexibility": diagnostics[
                    "avoidable_export_exceeds_flexibility"
                ],
                "grid_reference_exceeded": diagnostics[
                    "grid_reference_exceeded"
                ],
                "net_consumption_reference_exceeded": diagnostics[
                    "net_consumption_reference_exceeded"
                ],
                "cost_reference_exceeded": diagnostics[
                    "cost_reference_exceeded"
                ],
                "emission_reference_exceeded": diagnostics[
                    "emission_reference_exceeded"
                ],
                "terminal": diagnostics["terminal"],
                "terminal_soc_penalty": diagnostics[
                    "terminal_soc_penalty"
                ],
                "pv_to_load": pv_to_load,
                "pv_to_battery": pv_to_battery,
                "pv_self_consumed": pv_self_consumed,
                "total_demand": total_demand,
                "net_baseline": net_baseline,
                "baseline_grid_import": baseline_grid_import,
                "baseline_cost": baseline_cost,
                "positive_baseline_cost": positive_baseline_cost,
                "baseline_emission": baseline_emission,
                "positive_baseline_emission": positive_baseline_emission,
            }
        )

        observations = next_observations

        if done:
            break

    if not records:
        raise RuntimeError(
            "Evaluation completed without recording any transitions."
        )

    timeseries = pd.DataFrame(records)
    cost_function = CostFunction()

    accumulated_reward = float(timeseries["reward"].sum())
    total_grid_import = float(timeseries["grid_import"].sum())
    total_grid_export = float(timeseries["grid_export"].sum())
    total_unavoidable_export = float(
        timeseries["unavoidable_export"].sum()
    )
    total_avoidable_export = float(timeseries["avoidable_export"].sum())
    total_grid_penalty = float(timeseries["raw_grid_penalty"].sum())
    total_electricity_cost = float(timeseries["raw_cost"].sum())
    total_carbon_emission = float(timeseries["raw_emission"].sum())
    total_weighted_grid = float(timeseries["weighted_grid"].sum())
    total_weighted_cost = float(timeseries["weighted_cost"].sum())
    total_weighted_emission = float(
        timeseries["weighted_emission"].sum()
    )
    total_terminal_penalty = float(
        timeseries["terminal_soc_penalty"].sum()
    )

    baseline_grid_import = float(timeseries["baseline_grid_import"].sum())
    baseline_cost_total = float(timeseries["positive_baseline_cost"].sum())
    baseline_emission_total = float(
        timeseries["positive_baseline_emission"].sum()
    )

    total_pv_generation = float(timeseries["pv_generation"].sum())
    total_pv_self_consumed = float(timeseries["pv_self_consumed"].sum())
    total_demand = float(timeseries["total_demand"].sum())

    pv_self_consumption_rate = (
        100.0
        * total_pv_self_consumed
        / (total_pv_generation + EPSILON)
    )
    self_sufficiency_rate = (
        100.0 * total_pv_self_consumed / (total_demand + EPSILON)
    )

    net_consumption = timeseries["net_consumption"].to_numpy(dtype=float)
    net_baseline_array = timeseries["net_baseline"].to_numpy(dtype=float)
    costs = timeseries["raw_cost"].to_numpy(dtype=float)
    costs_baseline = timeseries["positive_baseline_cost"].to_numpy(
        dtype=float
    )
    emissions = timeseries["raw_emission"].to_numpy(dtype=float)
    emissions_baseline = timeseries["positive_baseline_emission"].to_numpy(
        dtype=float
    )

    initial_soc = float(timeseries["pre_action_soc"].iloc[0])
    final_soc = float(timeseries["post_action_soc"].iloc[-1])
    soc_difference = final_soc - initial_soc
    absolute_soc_difference = abs(soc_difference)

    mean_absolute_action = float(timeseries["action"].abs().mean())
    action_sign_changes = int(
        (
            np.sign(timeseries["action"])
            .diff()
            .fillna(0.0)
            != 0.0
        ).sum()
    )

    reference_diagnostics = calculate_reference_diagnostics(timeseries)
    print_reference_diagnostics(reference_diagnostics)

    kpis = {
        "building": building.name,
        "evaluation_time_steps": int(len(timeseries)),
        "accumulated_reward": accumulated_reward,
        "total_grid_import_kwh": total_grid_import,
        "total_grid_export_kwh": total_grid_export,
        "total_unavoidable_export_kwh": total_unavoidable_export,
        "total_avoidable_export_kwh": total_avoidable_export,
        "total_grid_penalty_kwh": total_grid_penalty,
        "total_electricity_cost": total_electricity_cost,
        "total_carbon_emission_kg": total_carbon_emission,
        "baseline_grid_import_kwh": baseline_grid_import,
        "baseline_electricity_cost": baseline_cost_total,
        "baseline_carbon_emission_kg": baseline_emission_total,
        "grid_import_ratio": safe_ratio(
            total_grid_import,
            baseline_grid_import,
        ),
        "electricity_cost_ratio": safe_ratio(
            total_electricity_cost,
            baseline_cost_total,
        ),
        "carbon_emission_ratio": safe_ratio(
            total_carbon_emission,
            baseline_emission_total,
        ),
        "total_weighted_grid_penalty": total_weighted_grid,
        "total_weighted_cost_penalty": total_weighted_cost,
        "total_weighted_emission_penalty": total_weighted_emission,
        "total_terminal_soc_penalty": total_terminal_penalty,
        "initial_soc": initial_soc,
        "final_soc": final_soc,
        "soc_difference": soc_difference,
        "absolute_soc_difference": absolute_soc_difference,
        "pv_self_consumption_rate_percent": pv_self_consumption_rate,
        "self_sufficiency_rate_percent": self_sufficiency_rate,
        "mean_absolute_action": mean_absolute_action,
        "action_sign_changes": action_sign_changes,
        "Z Zero Net Energy": safe_ratio(
            cost_function.zero_net_energy(net_consumption)[-1],
            cost_function.zero_net_energy(net_baseline_array)[-1],
        ),
        "C Price Cost": safe_ratio(
            cost_function.price(costs)[-1],
            cost_function.price(costs_baseline)[-1],
        ),
        "G Emission Cost": safe_ratio(
            cost_function.carbon_emissions(emissions)[-1],
            cost_function.carbon_emissions(emissions_baseline)[-1],
        ),
        "R Ramping": safe_ratio(
            cost_function.ramping(net_consumption)[-1],
            cost_function.ramping(net_baseline_array)[-1],
        ),
        "1-L Load Factor": safe_ratio(
            cost_function.load_factor(net_consumption)[-1],
            cost_function.load_factor(net_baseline_array)[-1],
        ),
        "D Net Electricity Consumption": safe_ratio(
            cost_function.net_electricity_consumption(net_consumption)[-1],
            cost_function.net_electricity_consumption(net_baseline_array)[-1],
        ),
        "grid_weight": reward_function.grid_weight,
        "cost_weight": reward_function.cost_weight,
        "emission_weight": reward_function.emission_weight,
        "avoidable_export_reference": (
            reward_function.avoidable_export_reference
        ),
        "net_consumption_reference": (
            reward_function.net_consumption_reference
        ),
        "cost_reference": reward_function.cost_reference,
        "emission_reference": reward_function.emission_reference,
        "soc_min": reward_function.soc_min,
        "soc_max": reward_function.soc_max,
        "export_tariff": reward_function.export_tariff,
        "terminal_soc_weight": reward_function.terminal_soc_weight,
        "epsilon": reward_function.epsilon,
    }
    kpis.update(reference_diagnostics)

    print("\nEvaluation KPIs:")

    for name, value in kpis.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            print("{:<50}: {:.6f}".format(name, float(value)))
        else:
            print("{:<50}: {}".format(name, value))

    elapsed_time = time.time() - start_time
    print(
        "\nEvaluation complete. Accumulated reward: {:.6f}".format(
            accumulated_reward
        )
    )
    print("Total evaluation time: {:.2f} seconds".format(elapsed_time))

    if save:
        result_directory = os.path.join(root_dir, simulation_id)
        os.makedirs(result_directory, exist_ok=True)

        timeseries.to_csv(
            os.path.join(result_directory, "timeseries.csv"),
            index=False,
        )
        pd.DataFrame([kpis]).to_csv(
            os.path.join(result_directory, "kpis.csv"),
            index=False,
        )

        reward_configuration = {
            "reward_type": reward_function.__class__.__name__,
            "building": building.name,
            "grid_weight": reward_function.grid_weight,
            "cost_weight": reward_function.cost_weight,
            "emission_weight": reward_function.emission_weight,
            "avoidable_export_reference": (
                reward_function.avoidable_export_reference
            ),
            "net_consumption_reference": (
                reward_function.net_consumption_reference
            ),
            "cost_reference": reward_function.cost_reference,
            "emission_reference": reward_function.emission_reference,
            "soc_min": reward_function.soc_min,
            "soc_max": reward_function.soc_max,
            "export_tariff": reward_function.export_tariff,
            "terminal_soc_weight": reward_function.terminal_soc_weight,
            "epsilon": reward_function.epsilon,
            "initial_soc": initial_soc,
            "final_soc": final_soc,
            "soc_difference": soc_difference,
            "absolute_soc_difference": absolute_soc_difference,
        }
        reward_configuration.update(reference_diagnostics)

        safe_json_dump(
            reward_configuration,
            os.path.join(result_directory, "reward_configuration.json"),
        )

        print("Results saved to: {}".format(result_directory))

    return accumulated_reward, kpis


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        prog="citylearn_policy_evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained TD3 agent pickle file.",
    )
    parser.add_argument(
        "--simulation_id",
        type=str,
        default="evaluation",
        help="Name of the evaluation result directory.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="..",
        help="Root directory for evaluation results.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Required: evaluate deterministic post-exploration actions.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save evaluation results.",
    )

    arguments = parser.parse_args()

    if not arguments.deterministic:
        raise ValueError(
            "Evaluation must be invoked with --deterministic."
        )

    model_path = os.path.abspath(arguments.model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "Could not find trained model: {}".format(model_path)
        )

    print("\nModel file being loaded:")
    print(model_path)
    print("Model last modified:", time.ctime(os.path.getmtime(model_path)))

    evaluation_schema, _ = load_eval_schema_from_settings()
    environment = SocFeasibleCityLearnEnv(evaluation_schema)

    with open(model_path, "rb") as stream:
        trained_model = pickle.load(stream)

    validate_model_environment_compatibility(trained_model, environment)

    evaluate(
        env=environment,
        model=trained_model,
        simulation_id=arguments.simulation_id,
        root_dir=arguments.root_dir,
        deterministic=arguments.deterministic,
        save=not arguments.no_save,
    )


if __name__ == "__main__":
    main()
