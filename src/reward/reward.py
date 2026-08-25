from typing import List
import numpy as np
from citylearn.reward_function import RewardFunction
import os 
import sys
from pathlib import Path
print(f"USING {os.path.join(*Path(os.path.dirname(__file__)).absolute().parts[0:])},{ os.path.basename(__file__)} ")
from typing import List
import numpy as np
from citylearn.reward_function import RewardFunction
from typing import List
import numpy as np
from citylearn.reward_function import RewardFunction


class NormCCEReward(RewardFunction):
    """Normalized avoidable-export, cost, and emission reward.
    The reward contains three normalized penalty terms:
        1. Avoidable grid export
        2. Grid electricity cost
        3. Operational carbon emissions
    The transition reward is:
        reward = -(
            grid_weight * normalized_avoidable_export
            + cost_weight * normalized_cost
            + emission_weight * normalized_emission
        ) + terminal_soc_penalty
    Notes
    -----
    The normalization references must be fixed before training and must
    remain identical across all buildings, TD3 configurations, seeds,
    local evaluations, and zero-shot transfer evaluations.
    The net-consumption reference is retained as a diagnostic reference.
    It is not included as a fourth reward objective.
    """
    def __init__(self,electricity_consumption: List[float] = None,**kwargs):
        """Initialize the normalized CCE reward."""
        super().__init__(electricity_consumption=electricity_consumption,**kwargs)

        self.grid_weight = float(self.kwargs["grid_weight"])
        self.cost_weight = float(self.kwargs["cost_weight"])
        self.emission_weight = float(self.kwargs["emission_weight"])

        self.avoidable_export_reference = float(self.kwargs["avoidable_export_reference"])
        self.net_consumption_reference = float(self.kwargs["net_consumption_reference"])
        self.cost_reference = float(self.kwargs["cost_reference"])
        self.emission_reference = float(self.kwargs["emission_reference"])

        self.soc_min = float(self.kwargs.get("soc_min",0.0))
        self.soc_max = float(self.kwargs.get("soc_max",1.0))
        self.export_tariff = float(self.kwargs.get("export_tariff",0.0))

        self.terminal_soc_weight = float(self.kwargs.get("terminal_soc_weight",0.0))
        self.epsilon = float(self.kwargs.get("epsilon",1e-8))

        # Tolerance used only for physical and reference checks.
        self.validation_tolerance = float(self.kwargs.get("validation_tolerance",1e-6))

        # When False, only the latest transition diagnostics are retained.
        # This avoids excessive memory use during long TD3 training.
        self.store_diagnostic_history = bool(self.kwargs.get("store_diagnostic_history",False))

        self._validate_parameters()
        self.initial_soc = None
        self.latest_diagnostics = {}
        self.diagnostic_history = []

    def _validate_parameters(self):
        """Validate reward weights and fixed parameters."""

        weights = np.asarray([self.grid_weight,self.cost_weight,self.emission_weight],dtype=float)
        references = np.asarray([self.avoidable_export_reference,self.net_consumption_reference,self.cost_reference,self.emission_reference],dtype=float)

        if not np.all(np.isfinite(weights)):
            raise ValueError("All NormCCEReward weights must be finite.")

        if np.any(weights < 0.0):raise ValueError("NormCCEReward weights must be non-negative.")

        if not np.isclose(
            weights.sum(),1.0,rtol=1e-6,atol=1e-6):
            raise ValueError(
                "NormCCEReward weights must sum to 1.0. "
                "Received sum: {:.8f}.".format(weights.sum()))

        if not np.all(np.isfinite(references)):
            raise ValueError(
                "All normalization references must be finite.")

        if np.any(references <= 0.0):
            raise ValueError(
                "All normalization references must be "
                "greater than zero.")

        if not (0.0 <= self.soc_min < self.soc_max <= 1.0):
            raise ValueError(
                "SOC limits must satisfy "
                "0 <= soc_min < soc_max <= 1.")

        if (not np.isfinite(self.export_tariff)or self.export_tariff < 0.0):
            raise ValueError(
                "Export tariff must be finite and non-negative.")

        if (not np.isfinite(self.terminal_soc_weight)or self.terminal_soc_weight < 0.0):
            raise ValueError(
                "Terminal SOC weight must be finite "
                "and non-negative.")

        if (not np.isfinite(self.epsilon)or self.epsilon <= 0.0):
            raise ValueError(
                "Epsilon must be finite and greater than zero.")

        if (not np.isfinite(self.validation_tolerance)or self.validation_tolerance <= 0.0):
            raise ValueError(
                "Validation tolerance must be finite "
                "and greater than zero.")

    def reset(self):
        """Reset episode-dependent reward state."""
        self.initial_soc = None
        self.latest_diagnostics = {}
        self.diagnostic_history = []

    def _get_transition_context(self):
        """Retrieve and validate the environment transition context."""
        context = self.kwargs.get("transition_context")

        if context is None:
            raise RuntimeError(
                "NormCCEReward requires transition_context "
                "from SocFeasibleCityLearnEnv.")

        required_values = [
            "pre_action_soc",
            "post_action_soc",
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
            "buy_price",
            "carbon_intensity",
            "terminal"]

        missing_values = [
            key
            for key in required_values
            if key not in context]

        if missing_values:
            raise KeyError(
                "Missing transition-context values: {}".format(missing_values))

        return context

    def _validate_transition_values(self,pre_action_soc,post_action_soc,load,pv_generation,base_net_demand,maximum_charge,maximum_discharge,feasible_lower,feasible_upper,projected_target,battery_consumption,net_consumption,grid_import,grid_export,unavoidable_export,avoidable_export,buy_price,carbon_intensity):
        """Validate transition values and physical limits."""

        transition_values = np.asarray([
                pre_action_soc,
                post_action_soc,
                load,
                pv_generation,
                base_net_demand,
                maximum_charge,
                maximum_discharge,
                feasible_lower,
                feasible_upper,
                projected_target,
                battery_consumption,
                net_consumption,
                grid_import,
                grid_export,
                unavoidable_export,
                avoidable_export,
                buy_price,
                carbon_intensity],dtype=float)

        if not np.all(np.isfinite(transition_values)):
            raise ValueError(
                "Transition context contains NaN or "
                "infinite values.")

        tolerance = self.validation_tolerance

        if not (self.soc_min - tolerance<= pre_action_soc<= self.soc_max + tolerance):
            raise ValueError(
                "Pre-action SOC is outside the configured limits. "
                "SOC={:.8f}, limits=[{:.8f}, {:.8f}].".format(pre_action_soc,self.soc_min,self.soc_max))

        if not (self.soc_min - tolerance<= post_action_soc<= self.soc_max + tolerance):
            raise ValueError(
                "Post-action SOC is outside the configured limits. "
                "SOC={:.8f}, limits=[{:.8f}, {:.8f}].".format(post_action_soc,self.soc_min,self.soc_max))

        if maximum_charge < -tolerance:
            raise ValueError("maximum_charge_energy must be non-negative.")

        if maximum_discharge < -tolerance:
            raise ValueError("maximum_discharge_energy must be non-negative.")

        if feasible_lower > feasible_upper + tolerance:
            raise ValueError(
                "Feasible grid limits are reversed. "
                "Lower={:.8f}, upper={:.8f}.".format(feasible_lower,feasible_upper))

        if not (feasible_lower - tolerance <= projected_target <= feasible_upper + tolerance):
            raise ValueError(
                "Projected grid target lies outside the "
                "feasible interval. "
                "Lower={:.8f}, target={:.8f}, "
                "upper={:.8f}.".format(feasible_lower,projected_target,feasible_upper))

        nonnegative_values = {"load":load,"pv_generation":pv_generation,"grid_import":grid_import,"grid_export":grid_export,"unavoidable_export":unavoidable_export,"avoidable_export":avoidable_export,"buy_price":buy_price,"carbon_intensity":carbon_intensity}

        for value_name, value in nonnegative_values.items():
            if value < -tolerance:
                raise ValueError(
                    "{} must be non-negative. "
                    "Received {:.8f}.".format(value_name,value))

    def calculate(self) -> np.ndarray:
        """Calculate the normalized multi-objective reward."""
        context = self._get_transition_context()

        pre_action_soc = float(context["pre_action_soc"])
        post_action_soc = float(context["post_action_soc"])
        load = float(context["load"])
        pv_generation = float(context["pv_generation"])
        base_net_demand = float(context["base_net_demand"])
        maximum_charge = float(context["maximum_charge_energy"])
        maximum_discharge = float(context["maximum_discharge_energy"])
        feasible_lower = float(context["feasible_lower_grid_limit"])
        feasible_upper = float(context["feasible_upper_grid_limit"])
        projected_target = float(context["projected_grid_target"])
        battery_consumption = float(context["battery_consumption"]) 
        net_consumption = float(context["net_consumption"])
        grid_import = float(context["grid_import"])
        grid_export = float(context["grid_export"])
        unavoidable_export = float(context["unavoidable_export"])
        avoidable_export = float(context["avoidable_export"])
        buy_price = float(context["buy_price"])
        carbon_intensity = float(context["carbon_intensity"])
        terminal = bool(context["terminal"])

        self._validate_transition_values(
            pre_action_soc=pre_action_soc,
            post_action_soc=post_action_soc,
            load=load,
            pv_generation=pv_generation,
            base_net_demand=base_net_demand,
            maximum_charge=maximum_charge,
            maximum_discharge=maximum_discharge,
            feasible_lower=feasible_lower,
            feasible_upper=feasible_upper,
            projected_target=projected_target,
            battery_consumption=battery_consumption,
            net_consumption=net_consumption,
            grid_import=grid_import,
            grid_export=grid_export,
            unavoidable_export=unavoidable_export,
            avoidable_export=avoidable_export,
            buy_price=buy_price,
            carbon_intensity=carbon_intensity)

        if self.initial_soc is None:
            self.initial_soc = pre_action_soc

        reconstructed_net = (base_net_demand + battery_consumption)

        if not np.isclose(reconstructed_net,net_consumption,rtol=1e-6,atol=1e-6):
            raise ValueError(
                "Net-consumption identity failed. "
                "CityLearn={:.8f}, reconstructed={:.8f}.".format(
                    net_consumption,
                    reconstructed_net))

        expected_grid_import = max(net_consumption,0.0)
        expected_grid_export = max(-net_consumption,0.0)
        expected_unavoidable_export = max(-projected_target,0.0)
        expected_avoidable_export = max(expected_grid_export - expected_unavoidable_export,0.0)

        if not np.isclose(grid_import,expected_grid_import,rtol=1e-6,atol=1e-8):
            raise ValueError(
                "Grid-import consistency check failed. "
                "Context={:.8f}, expected={:.8f}.".format(grid_import,expected_grid_import))

        if not np.isclose(grid_export,expected_grid_export,rtol=1e-6,atol=1e-8):
            raise ValueError(
                "Grid-export consistency check failed. "
                "Context={:.8f}, expected={:.8f}.".format(grid_export,expected_grid_export))

        if not np.isclose(unavoidable_export,expected_unavoidable_export,rtol=1e-6,atol=1e-8):
            raise ValueError(
                "Unavoidable-export consistency check failed. "
                "Context={:.8f}, expected={:.8f}.".format(unavoidable_export,expected_unavoidable_export))

        if not np.isclose(avoidable_export,expected_avoidable_export,rtol=1e-6,atol=1e-8):
            raise ValueError(
                "Avoidable-export consistency check failed. "
                "Context={:.8f}, expected={:.8f}.".format(avoidable_export,expected_avoidable_export))

        raw_grid_penalty = avoidable_export
        raw_cost = (buy_price * grid_import- self.export_tariff * grid_export)
        raw_emission = (carbon_intensity* grid_import)

        normalized_grid = (raw_grid_penalty/ (self.avoidable_export_reference + self.epsilon))
        # Diagnostic only. Not included in the reward.
        normalized_grid_import = (grid_import/ (self.net_consumption_reference + self.epsilon))
        normalized_cost = (raw_cost / (self.cost_reference+ self.epsilon))
        normalized_emission = (raw_emission / (self.emission_reference+ self.epsilon))

        reference_tolerance = (self.validation_tolerance)
        grid_reference_exceeded = bool(normalized_grid> 1.0 + reference_tolerance)
        net_consumption_reference_exceeded = bool(normalized_grid_import> 1.0 + reference_tolerance)
        cost_reference_exceeded = bool(raw_cost >= 0.0and normalized_cost> 1.0 + reference_tolerance)
        emission_reference_exceeded = bool(normalized_emission> 1.0 + reference_tolerance)

        feasible_grid_span = max(feasible_upper - feasible_lower, 0.0)
        if feasible_grid_span > self.epsilon:avoidable_export_fraction_of_flexibility = (avoidable_export / feasible_grid_span)
        else:
            avoidable_export_fraction_of_flexibility = 0.0
        avoidable_export_exceeds_flexibility = bool(avoidable_export> (feasible_grid_span+ reference_tolerance ))

        weighted_grid = (self.grid_weight* normalized_grid)
        weighted_cost = (self.cost_weight* normalized_cost)
        weighted_emission = (self.emission_weight* normalized_emission)
        terminal_penalty = 0.0

        if (terminal and self.terminal_soc_weight > 0.0):
            normalized_terminal_soc_difference = (abs( post_action_soc - self.initial_soc ) / ( self.soc_max  - self.soc_min + self.epsilon))
            terminal_penalty = ( -self.terminal_soc_weight * normalized_terminal_soc_difference)

        reward_value = -(weighted_grid + weighted_cost + weighted_emission) + terminal_penalty

        diagnostic_values = np.asarray(
            [
                pre_action_soc,
                post_action_soc,
                load,
                pv_generation,
                base_net_demand,
                maximum_charge,
                maximum_discharge,
                feasible_lower,
                feasible_upper,
                projected_target,
                battery_consumption,
                net_consumption,
                grid_import,
                grid_export,
                unavoidable_export,
                avoidable_export,
                raw_grid_penalty,
                raw_cost,
                raw_emission,
                normalized_grid,
                normalized_grid_import,
                normalized_cost,
                normalized_emission,
                weighted_grid,
                weighted_cost,
                weighted_emission,
                feasible_grid_span,
                avoidable_export_fraction_of_flexibility,
                terminal_penalty,
                reward_value,
                self.avoidable_export_reference,
                self.net_consumption_reference,
                self.cost_reference,
                self.emission_reference ],dtype=float)
        
        if not np.all(np.isfinite( diagnostic_values )):
            raise ValueError( "NormCCEReward generated NaN or infinite values.")

        expected_reward = -( weighted_grid + weighted_cost + weighted_emission) + terminal_penalty

        if not np.isclose(reward_value,expected_reward,rtol=1e-10,atol=1e-10):
            raise AssertionError("NormCCEReward identity check failed.")

        self.latest_diagnostics = {
            "pre_action_soc":pre_action_soc,
            "post_action_soc":post_action_soc,
            "initial_episode_soc":self.initial_soc,
            "load":load,
            "pv_generation":pv_generation,
            "base_net_demand":base_net_demand,
            "maximum_charge_energy":maximum_charge,
            "maximum_discharge_energy":maximum_discharge,
            "feasible_lower_grid_limit":feasible_lower,
            "feasible_upper_grid_limit":feasible_upper,
            "projected_grid_target":projected_target,
            "battery_consumption":battery_consumption,
            "net_consumption":net_consumption,
            "grid_import":grid_import,
            "grid_export":grid_export,
            "unavoidable_export":unavoidable_export,
            "avoidable_export":avoidable_export,
            "raw_grid_penalty":raw_grid_penalty,
            "raw_cost":raw_cost,
            "raw_emission":raw_emission,
            "avoidable_export_reference":self.avoidable_export_reference,
            "net_consumption_reference":self.net_consumption_reference,
            "cost_reference":self.cost_reference,
            "emission_reference":self.emission_reference,
            "normalized_grid":normalized_grid,
            "normalized_grid_import":normalized_grid_import,
            "normalized_cost":normalized_cost,
            "normalized_emission":normalized_emission,
            "weighted_grid":weighted_grid,
            "weighted_cost":weighted_cost,
            "weighted_emission":weighted_emission,
            "feasible_grid_span":feasible_grid_span,
            "avoidable_export_fraction_of_flexibility":avoidable_export_fraction_of_flexibility,
            "avoidable_export_exceeds_flexibility":avoidable_export_exceeds_flexibility,
            "grid_reference_exceeded":grid_reference_exceeded,
            "net_consumption_reference_exceeded":net_consumption_reference_exceeded,
            "cost_reference_exceeded":cost_reference_exceeded,
            "emission_reference_exceeded":emission_reference_exceeded,
            "terminal":terminal,
            "terminal_soc_penalty":terminal_penalty,
            "reward":reward_value}

        if self.store_diagnostic_history:self.diagnostic_history.append(self.latest_diagnostics.copy())
        return np.asarray([reward_value],dtype=float)

    
#============================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------------------
#============================================================================================================================================================


class AdditiveSolarPenaltyReward(RewardFunction):
    def __init__(self, electricity_consumption: List[float] = None, **kwargs):
        super().__init__(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]: #array rakae

        carbon_emission = (np.array(self.carbon_emission)*self.kwargs['carbon_emission_weight'])**self.kwargs['carbon_emission_exponent']
        electricity_price = (np.array(self.electricity_price)*self.kwargs['electricity_price_weight'])**self.kwargs['electricity_price_exponent']
        soc = self.kwargs.get('electrical_storage_soc', np.array([0.0]*self.agent_count))
        reward = -(1.0 + np.sign(electricity_price)*soc)*abs(carbon_emission + electricity_price)
        # print('soc: ',soc, ' ele_pric: ',electricity_price, ' carbon_emission: ', carbon_emission,'reward: ',reward)
        return reward

class consumption(RewardFunction):
    # marlisa + p7 citylearn
    def __init__(self, electricity_consumption: List[float] = None, **kwargs):
        super().__init__(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        # print("data type: ", type(self.electricity_consumption), self.electricity_consumption)
        cons = -(max(0, max(self.electricity_consumption)))**self.kwargs['exp']
        reward = np.array([cons*self.kwargs['weight']])
        # print(reward)
        return reward

class flattning(RewardFunction):
    # from all papers
    def __init__(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        electricity_consumption = (np.array(self.electricity_consumption)*self.kwargs['weight'])**self.kwargs['exp']
        reward = -(electricity_consumption)
        # print('consumption:',self.electricity_consumption, 'reward: ',reward)
        return reward

class price(RewardFunction):
    # our own
    def _init_(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        electricity_price = (np.array(self.electricity_price)*self.kwargs['weight'])**self.kwargs['exp']
        reward = -(electricity_price)
        # print('price: ', electricity_price, 'reward: ',reward)
        return reward

class price_consumption(RewardFunction):
    # 
    def _init_(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        electricity_price = (np.array(self.electricity_price)*self.kwargs['electricity_price_weight'])**self.kwargs['electricity_price_exponent']
        electricity_consumption = (np.array(self.electricity_consumption)*self.kwargs['electricity_consumption_weight'])**self.kwargs['electricity_consumption_exponent']
        reward = -(electricity_price + electricity_consumption)
        return reward

class NZE(RewardFunction):
    # 
    def _init_(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        soc = self.kwargs.get('electrical_storage_soc', np.array([0.0]*self.agent_count))
        penalty = 1+np.sign(self.electricity_consumption)*soc
        reward = - penalty*abs(self.electricity_consumption[0])**self.kwargs['abs_exponent']
        return reward

class NZE2(RewardFunction):
    # 
    def _init_(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        soc = self.kwargs.get('electrical_storage_soc', np.array([0.0]*self.agent_count))
        penalty = 1+np.sign(self.electricity_consumption)*soc
        reward = - penalty*abs(self.electricity_consumption[0])**self.kwargs['abs_exponent']
        return reward

class NZE3(RewardFunction):
    # 
    def _init_(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        soc = self.kwargs.get('electrical_storage_soc', np.array([0.0]*self.agent_count))
        penalty = 1+np.sign(self.electricity_consumption)*soc
        reward = - penalty*abs(self.electricity_consumption[0])**self.kwargs['abs_exponent']
        return reward

class Price_base_penalty(RewardFunction):
    # 
    def _init_(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        price = self.kwargs.get('electricity_pricing', np.array([0.0]*self.agent_count))
        soc = self.kwargs.get('electrical_storage_soc', np.array([0.0]*self.agent_count))
        penalty = 1+np.sign(price-0.3)*soc
        reward= - penalty*abs(self.electricity_consumption[0])**self.kwargs['exponent']
        return reward

class Price_base_penalty2(RewardFunction):
    # 
    def _init_(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        price = self.kwargs.get('electricity_pricing', np.array([0.0]*self.agent_count))
        soc = self.kwargs.get('electrical_storage_soc', np.array([0.0]*self.agent_count))
        penalty = 1+np.sign(price-0.3)*soc
        reward= - penalty*abs(self.electricity_consumption[0])**self.kwargs['exponent']
        return reward

class Price_base_penalty3(RewardFunction):
    # 
    def _init_(self, electricity_consumption: List[float] = None, **kwargs):
        super()._init_(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        price = self.kwargs.get('electricity_pricing', np.array([0.0]*self.agent_count))
        soc = self.kwargs.get('electrical_storage_soc', np.array([0.0]*self.agent_count))
        penalty = 1+np.sign(price-0.3)*soc
        reward= - penalty*abs(self.electricity_consumption[0])**self.kwargs['exponent']
        return reward
