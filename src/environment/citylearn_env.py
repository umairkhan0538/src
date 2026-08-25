from typing import Any, Dict, List

import numpy as np

from citylearn.citylearn import CityLearnEnv


class SocFeasibleCityLearnEnv(CityLearnEnv):
    def __init__(self, schema, **kwargs):
        super().__init__(schema, **kwargs)
        if len(self.buildings) != 1:
            raise ValueError("SocFeasibleCityLearnEnv requires exactly one building.")
    @staticmethod
    def _get_normalized_soc(battery) -> float:
        capacity = float(battery.capacity)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("Battery capacity must be finite and greater than zero.")
        normalized_soc = float(battery.soc_init) / capacity
        return float(np.clip(normalized_soc, 0.0, 1.0))
    def _get_pre_action_context(self) -> Dict[str, Any]:
        battery = self.buildings[0].electrical_storage
        capacity = float(battery.capacity)
        available_soc_energy = float(battery.soc_init)
        pre_action_soc = self._get_normalized_soc(battery)
        soc_min = float(self.reward_function.kwargs.get("soc_min", 0.0))
        soc_max = float(self.reward_function.kwargs.get("soc_max", 1.0))
        if not 0.0 <= soc_min < soc_max <= 1.0:
            raise ValueError("SOC limits must satisfy 0 <= soc_min < soc_max <= 1.")
        maximum_charge_power = max(float(battery.get_max_input_power()), 0.0,)
        maximum_discharge_power = max(float(battery.get_max_output_power()), 0.0,)

        charge_efficiency = float(battery.get_current_efficiency(maximum_charge_power))
        discharge_efficiency = float(battery.get_current_efficiency(-maximum_discharge_power))

        if not np.isfinite(charge_efficiency) or charge_efficiency <= 0.0:
            raise ValueError("Battery charging efficiency must be finite and positive.")
        if not np.isfinite(discharge_efficiency) or discharge_efficiency <= 0.0:
            raise ValueError("Battery discharging efficiency must be finite and positive.")

        maximum_soc_energy = soc_max * capacity
        minimum_soc_energy = soc_min * capacity

        charge_energy_by_soc = max(maximum_soc_energy - available_soc_energy,0.0,) / charge_efficiency
        discharge_energy_by_soc = max(available_soc_energy - minimum_soc_energy,0.0,) * discharge_efficiency
        maximum_charge_energy = min(maximum_charge_power,charge_energy_by_soc,)
        maximum_discharge_energy = min(maximum_discharge_power,discharge_energy_by_soc,)

        context = {
            "action_time_step": int(self.time_step),
            "pre_action_soc": pre_action_soc,
            "pre_action_soc_energy": available_soc_energy,
            "pre_action_capacity": capacity,
            "soc_min": soc_min,
            "soc_max": soc_max,
            "maximum_charge_power": maximum_charge_power,
            "maximum_discharge_power": maximum_discharge_power,
            "maximum_charge_energy": maximum_charge_energy,
            "maximum_discharge_energy": maximum_discharge_energy,
            "charge_efficiency": charge_efficiency,
            "discharge_efficiency": discharge_efficiency,
        }
        self._validate_finite_values(context)
        return context
    
    def _get_post_action_context(self,pre_action_context: Dict[str, Any],) -> Dict[str, Any]:
        building = self.buildings[0]
        battery = building.electrical_storage
        reward_time_step = int(self.time_step)
        post_action_capacity = float(battery.capacity)

        if not np.isfinite(post_action_capacity) or post_action_capacity <= 0.0:
            raise ValueError("Post-action battery capacity must be finite and greater than zero.")

        post_action_soc_energy = float(battery.soc[-1])
        post_action_soc = post_action_soc_energy / post_action_capacity

        load = float(building.non_shiftable_load_demand[reward_time_step])
        citylearn_pv = float(building.solar_generation[reward_time_step])
        positive_pv = max(-citylearn_pv, 0.0)

        battery_consumption = float(building.electrical_storage_electricity_consumption[reward_time_step])
        net_consumption = float(building.net_electricity_consumption[reward_time_step])
        electricity_price = float(building.pricing.electricity_pricing[reward_time_step])
        carbon_intensity = float(building.carbon_intensity.carbon_intensity[reward_time_step])

        base_net_demand = load - positive_pv
        reconstructed_net_consumption = (base_net_demand + battery_consumption)

        if not np.isclose(reconstructed_net_consumption,net_consumption,rtol=1e-6,atol=1e-6,):
            raise ValueError(
                "Net-consumption identity failed at reward time step "
                f"{reward_time_step}. "
                f"load={load:.8f}, "
                f"PV={positive_pv:.8f}, "
                f"battery={battery_consumption:.8f}, "
                f"CityLearn net={net_consumption:.8f}, "
                f"reconstructed net={reconstructed_net_consumption:.8f}.")

        feasible_lower_grid_limit = (base_net_demand- pre_action_context["maximum_discharge_energy"])
        feasible_upper_grid_limit = (base_net_demand + pre_action_context["maximum_charge_energy"])

        projected_grid_target = self._project_zero(feasible_lower_grid_limit,feasible_upper_grid_limit,)

        grid_import = max(net_consumption, 0.0)
        grid_export = max(-net_consumption, 0.0)
        unavoidable_export = max(-projected_grid_target, 0.0)
        avoidable_export = max(grid_export - unavoidable_export,0.0,)

        context = {
            **pre_action_context,
            "reward_time_step": reward_time_step,
            "post_action_capacity": post_action_capacity,
            "post_action_soc_energy": post_action_soc_energy,
            "post_action_soc": float(np.clip(post_action_soc, 0.0, 1.0)),
            "load": load,
            "citylearn_solar_generation": citylearn_pv,
            "pv_generation": positive_pv,
            "battery_consumption": battery_consumption,
            "base_net_demand": base_net_demand,
            "reconstructed_net_consumption": reconstructed_net_consumption,
            "net_consumption": net_consumption,
            "feasible_lower_grid_limit": feasible_lower_grid_limit,
            "feasible_upper_grid_limit": feasible_upper_grid_limit,
            "projected_grid_target": projected_grid_target,
            "grid_import": grid_import,
            "grid_export": grid_export,
            "unavoidable_export": unavoidable_export,
            "avoidable_export": avoidable_export,
            "buy_price": electricity_price,
            "carbon_intensity": carbon_intensity,
            "terminal": bool(self.done),
        }

        self._validate_finite_values(context)
        self._validate_battery_response(context)
        return context

    @staticmethod
    def _project_zero(lower_limit: float, upper_limit: float) -> float:
        if lower_limit > upper_limit:
            raise ValueError("Feasible lower grid limit cannot exceed the feasible upper grid limit.")

        if lower_limit > 0.0:
            return lower_limit
        if upper_limit < 0.0:
            return upper_limit
        return 0.0

    @staticmethod
    def _validate_finite_values(context: Dict[str, Any]) -> None:
        numerical_values = [value for key, value in context.items() if key != "terminal" and isinstance(value,(int, float, np.integer, np.floating), )]

        if not numerical_values:
            raise ValueError("The transition context contains no numerical values.")
        if not np.all(np.isfinite(np.asarray(numerical_values, dtype=float))):
            raise ValueError("The reward transition context contains NaN or infinite values.")
    @staticmethod
    def _validate_battery_response(context: Dict[str, Any]) -> None:
        soc_change = (context["post_action_soc"]- context["pre_action_soc"])
        battery_consumption = context["battery_consumption"]

        soc_tolerance = 1e-5
        energy_tolerance = 1e-6
        if (soc_change > soc_tolerance and battery_consumption <= energy_tolerance):
            raise ValueError(
                "Battery timing inconsistency: SOC increased without "
                "positive realized battery charging. "
                f"action_time_step={context['action_time_step']}, "
                f"reward_time_step={context['reward_time_step']}, "
                f"pre_SOC={context['pre_action_soc']:.8f}, "
                f"post_SOC={context['post_action_soc']:.8f}, "
                f"battery_consumption={battery_consumption:.8f}.")

        if (soc_change < -soc_tolerance and battery_consumption >= -energy_tolerance):
            raise ValueError(
                "Battery timing inconsistency: SOC decreased without "
                "negative realized battery discharging. "
                f"action_time_step={context['action_time_step']}, "
                f"reward_time_step={context['reward_time_step']}, "
                f"pre_SOC={context['pre_action_soc']:.8f}, "
                f"post_SOC={context['post_action_soc']:.8f}, "
                f"battery_consumption={battery_consumption:.8f}."
            )

    def step(self, actions: List[List[float]]):
        pre_action_context = self._get_pre_action_context()
        parsed_actions = self._CityLearnEnv__parse_actions(actions)

        for building, building_actions in zip(self.buildings, parsed_actions,):
            building.apply_actions(**building_actions)

        self.next_time_step()
        transition_context = self._get_post_action_context(pre_action_context)
        self.reward_function.kwargs["transition_context"] = transition_context

        reward = self.get_reward()
        self._CityLearnEnv__rewards.append(reward)

        return (self.observations,reward,self.done,self.get_info(),)

    def reset(self):
        observations = super().reset()
        if hasattr(self.reward_function, "reset"):
            self.reward_function.reset()
        return observations
