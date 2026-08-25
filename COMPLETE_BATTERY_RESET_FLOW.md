# Battery Reset Flow - Complete Lifecycle with Evidence

## Executive Summary
This document tracks the **complete battery reset flow** across multiple episodes with evidence from every file involved. Battery resets to `initial_soc` at episode start through a cascading sequence.

---

## 📊 High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SIMULATOR STARTS (simulate.py)                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ├─→ Load Environment: CityLearnEnv(schema)
                           │
                           └─→ Load Agent: env.load_agent()
                                │
                                ├─→ Episode 1
                                ├─→ Episode 2
                                ├─→ Episode 3
                                └─→ Episode 30 (if 30 episodes configured)


FOR EACH EPISODE:
┌────────────────────────────────────────────────────────────────┐
│              EPISODE START (simulator.py : Line 48-52)          │
│  for episode in trange(env.schema['episodes']):                │
│      observations = env.reset()  ◄────── RESET CALLED HERE    │
└─────────────┬──────────────────────────────────────────────────┘
              │
              └──→ RESET CASCADE BEGINS
                   │
                   ├─→ ┌────────────────────────────────────────┐
                   │   │ CityLearnEnv.reset() (citylearn.py:638) │
                   │   │ - Calls super().reset()                │
                   │   │ - Calls building.reset() for all       │
                   │   │ - Resets reward tracking              │
                   │   └────────────────┬─────────────────────┘
                   │                    │
                   │                    └──→ ┌─────────────────────────────────┐
                   │                        │ Building.reset() (building.py:829)│
                   │                        │ - Calls electrical_storage.reset()│
                   │                        │ - Calls cooling_storage.reset()  │
                   │                        │ - Calls heating_storage.reset()  │
                   │                        │ - Calls dhw_storage.reset()      │
                   │                        │ - Resets device states           │
                   │                        └────────────┬────────────────────┘
                   │                                     │
                   │                                     └──→ ┌───────────────────────────┐
                   │                                         │ StorageDevice.reset()      │
                   │                                         │ (energy_model.py:600)      │
                   │                                         │                           │
                   │                                         │ self.__soc = [initial_soc]│
                   │                                         │ self.__energy_balance = [0]│
                   │                                         │                           │
                   │                                         │ ✅ BATTERY RESET TO 0.0   │
                   │                                         └────────────┬──────────────┘
                   │                                                      │
                   └──────────────────────────────────────────────────────┘
                                          │
                        ┌─────────────────┴──────────────────┐
                        │                                    │
                        ▼                                    ▼
        ┌──────────────────────────────┐   ┌──────────────────────────────┐
        │    EPISODE EXECUTION         │   │   BATTERY STATE TRACKING     │
        │  (simulator.py : Line 55-78) │   │   Through Timesteps:         │
        │                              │   │                              │
        │ while not env.done:          │   │ Timestep 0: SOC = [0.0]      │
        │   action = agent.select()    │   │ Timestep 1: SOC = [0.0, 2.5] │
        │   obs, reward = env.step()   │   │ Timestep 2: SOC = [0.0, 2.5, │
        │   agent.add_to_buffer()      │   │                      4.1]    │
        │                              │   │ ...                          │
        │ metrics = env.evaluate()     │   │ Timestep N: SOC = [0.0, 2.5, │
        │                              │   │            4.1, 3.7, ...]    │
        └──────────────────────────────┘   └──────────────────────────────┘
                        │
                        │ Episode ends (env.done = True)
                        │ Battery SOC final value = 3.7 kWh
                        │ (This value is NOT carried to next episode)
                        │
                        ▼
        ┌──────────────────────────────┐
        │   NEXT EPISODE STARTS         │
        │   (simulator.py : Line 48)    │
        │                              │
        │   observations = env.reset() │
        │   ◄─────────────────────────  │ BATTERY RESETS AGAIN
        │                              │
        │   self.__soc = [0.0]         │
        │   (Fresh Start!)             │
        └──────────────────────────────┘
```

---

## 🔍 Complete File Evidence Map

### **File 1: `simulator.py` - Episode Loop Entry Point**

**Location:** `D:\Research\RL\Degradation analysis\201\src\simulate.py`

#### Evidence 1: Import CityLearn
```python
[Lines 11-15]
from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import write_json
from reward.reward_signals import extract_reward_signals
```
✅ **Evidence**: Uses CityLearn environment that handles reset

---

#### Evidence 2: Episode Loop & Reset Call
```python
[Lines 48-52]
# -------------------- Episode Loop --------------------
for episode in trange(env.schema['episodes'], desc="Simulating Episodes"):

    acc_reward = 0.0

    observations = env.reset()  ◄───── RESET CALLED HERE
```

✅ **Evidence**: At line 52, `env.reset()` is called at the START of each episode
- This is the **entry point** for the reset cascade
- Happens BEFORE the timestep loop starts (line 55)

---

#### Evidence 3: Timestep Loop (Episode Execution)
```python
[Lines 55-78]
# -------------------- Timestep Loop --------------------
while not env.done:

    actions = agents.select_actions(observations)
    next_obs, reward, _, _ = env.step(actions)
    print(f"citylearn reward : {reward}")
    # ... more steps ...
    
    observations = next_obs

    LOGGER.debug(
        f"Time step: {env.time_step}/{env.time_steps - 1}, "
        f"Episode: {episode}/{env.schema['episodes'] - 1}, "
        f"Actions: {actions}, Reward: {reward}"
    )
```

✅ **Evidence**: 
- While loop runs from time_step 0 to time_steps-1 (usually ~8760)
- Battery SOC evolves during this loop
- Loop exits when `env.done = True`
- Then next episode calls `env.reset()` again (line 48)

---

### **File 2: `citylearn.py` - Environment Reset**

**Location:** `c:\Users\Umair\Desktop\Research\citylearn\citylearn.py`

#### Evidence 4: CityLearnEnv.reset() Method
```python
[Lines 638-655]
def reset(self):
    r"""Reset `CityLearnEnv` to initial state.
    
    Returns
    -------
    observations: List[List[float]]
        :attr:`observations`. 
    """

    # object reset
    super().reset()

    for building in self.buildings:
        building.reset()  ◄───── CALLS BUILDING RESET

    # variable reset
    self.__rewards = [[]]
    self.__net_electricity_consumption = []
    self.__net_electricity_consumption_price = []
    self.__net_electricity_consumption_emission = []
    self.update_variables()

    return self.observations
```

✅ **Evidence**:
- Line 638: Method definition `def reset(self):`
- Line 647: `super().reset()` - resets parent Environment class
- Lines 649-650: **CRITICAL** - Iterates through all buildings and calls `building.reset()`
- Line 655: Returns observations for next episode

**Flow**: `env.reset()` → loops through buildings → calls `building.reset()` on each

---

### **File 3: `building.py` - Building Component Reset**

**Location:** `c:\Users\Umair\Desktop\Research\citylearn\building.py`

#### Evidence 5: Building.reset() Method
```python
[Lines 829-842]
def reset(self):
    r"""Reset `Building` to initial state."""

    # object reset
    super().reset()
    self.cooling_storage.reset()        ◄─── RESETS STORAGE 1
    self.heating_storage.reset()        ◄─── RESETS STORAGE 2
    self.dhw_storage.reset()            ◄─── RESETS STORAGE 3
    self.electrical_storage.reset()     ◄─── RESETS BATTERY HERE!
    self.cooling_device.reset()
    self.heating_device.reset()
    self.dhw_device.reset()
    self.pv.reset()

    # variable reset
    self.__cooling_electricity_consumption = []
    self.__heating_electricity_consumption = []
    self.__dhw_electricity_consumption = []
    self.__solar_generation = self.pv.get_generation(self.energy_simulation.solar_generation)*-1
    self.__net_electricity_consumption = []
    self.__net_electricity_consumption_emission = []
    self.__net_electricity_consumption_price = []
    self.update_variables()
```

✅ **Evidence**:
- Line 829: Method definition `def reset(self):`
- Line 833: Parent class reset
- **Line 837**: `self.electrical_storage.reset()` - **BATTERY RESET CALLED HERE**
- Also resets: cooling_storage, heating_storage, dhw_storage (line 835-836, 838)
- Lines 841-849: Resets tracking variables for electricity consumption

**Critical Line**: Line 837 - Battery (electrical_storage) reset

---

### **File 4: `energy_model.py` - Storage Device Reset (CRITICAL)**

**Location:** `c:\Users\Umair\Desktop\Research\citylearn\energy_model.py`

#### Evidence 6: StorageDevice Class Structure
```python
[Lines 440-500]
class StorageDevice(Device):
    def __init__(self, capacity: float, efficiency: float = None, 
                 loss_coefficient: float = None, initial_soc: float = None, 
                 efficiency_scaling: float = None, **kwargs):
        r"""Initialize `StorageDevice`.
        
        ...parameters...
        initial_soc : float, default: 0.0
            State of charge when `time_step` = 0. Must be >= 0 and < `capacity`.
        ...
        """
        
        self.efficiency_scaling = efficiency_scaling
        self.capacity = capacity
        self.loss_coefficient = loss_coefficient
        self.initial_soc = initial_soc  ◄─── INITIAL_SOC SET HERE (default 0.0)
        super().__init__(efficiency = efficiency, **kwargs)
```

✅ **Evidence**:
- Line 442-457: Constructor defines `initial_soc` parameter (default 0.0)
- Line 457: `self.initial_soc = initial_soc` stores the initial state of charge
- This value comes from schema.json configuration

---

#### Evidence 7: StorageDevice Reset Method (THE CRITICAL RESET)
```python
[Lines 600-605]
def reset(self):
    r"""Reset `StorageDevice` to initial state."""

    super().reset()
    self.__soc = [self.initial_soc]          ◄─── RESET TO INITIAL_SOC
    self.__energy_balance = [0.0]             ◄─── RESET ENERGY BALANCE
```

✅ **CRITICAL EVIDENCE**:
- Line 600: Method definition `def reset(self):`
- **Line 603**: `self.__soc = [self.initial_soc]` 
  - **SOC list is reset to contain ONLY the initial_soc value**
  - Default initial_soc = 0.0 kWh
  - This creates a NEW list, discarding all previous SOC history
- Line 604: Energy balance also reset to [0.0]

**This is where battery resets happen!**

---

#### Evidence 8: Battery Class (inherits from StorageDevice)
```python
[Lines 673-705]
class Battery(ElectricDevice, StorageDevice):
    def __init__(self, capacity: float, nominal_power: float, 
                 capacity_loss_coefficient: float = None, 
                 power_efficiency_curve: List[List[float]] = None, 
                 capacity_power_curve: List[List[float]] = None, **kwargs):
        
        r"""Initialize `Battery`.
        
        Parameters
        ----------
        capacity : float
            Maximum amount of energy the storage device can store in [kWh]
        ...
        """
        
        self.__efficiency_history = []
        self.__capacity_history = []
        super().__init__(capacity = capacity, nominal_power = nominal_power, **kwargs)
        self.capacity_loss_coefficient = capacity_loss_coefficient
        self.power_efficiency_curve = power_efficiency_curve
        self.capacity_power_curve = capacity_power_curve
```

✅ **Evidence**:
- Line 673: Battery inherits from StorageDevice
- Line 699: Calls `super().__init__()` which initializes StorageDevice
- Battery class REUSES StorageDevice.reset() method (no override)
- So Battery.reset() = StorageDevice.reset() which sets `self.__soc = [initial_soc]`

---

#### Evidence 9: SOC Property Access (Proof of Reset)
```python
[Lines 500-510]
@property
def soc(self) -> List[float]:
    r"""State of charge time series in [kWh]."""
    return self.__soc  ◄─── RETURNS SOC LIST

@property
def soc_init(self) -> float:
    r"""Latest state of charge after accounting for standby hourly lossses."""
    return self.__soc[-1]*(1 - self.loss_coefficient)
```

✅ **Evidence**:
- `soc` property returns the SOC list
- After reset(), this list contains ONLY [initial_soc]
- `soc_init` is latest SOC with losses applied

---

#### Evidence 10: Charge Method (Battery behavior during episode)
```python
[Lines 760-785]
def charge(self, energy: float):
    """Charges or discharges storage with respect to specified energy...
    
    Notes
    -----
    If charging, soc = min(`soc_init` + energy*`efficiency`, `max_input_power`, `capacity`)
    If discharging, soc = max(0, `soc_init` + energy/`efficiency`, `max_output_power`)
    """
    
    # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
    soc = min(self.soc_init + energy*self.efficiency, self.capacity) \
          if energy >= 0 else max(0, self.soc_init + energy/self.efficiency)
    
    self.__soc.append(soc)  ◄─── APPENDS NEW SOC VALUE
    self.__energy_balance.append(self.set_energy_balance())
```

✅ **Evidence**:
- Line 771: Calculates new SOC based on charge/discharge
- **Line 773**: `self.__soc.append(soc)` - APPENDS to SOC list during episode
- During an episode, SOC list grows: [0.0] → [0.0, 2.5] → [0.0, 2.5, 4.1] → ...
- When reset() is called, this entire list is cleared and reset to [0.0]

---

## 📈 Episode-by-Episode Battery State Tracking

### **Episode 1: Full Lifecycle**

```
TIME                    EVENT                           BATTERY STATE
─────────────────────────────────────────────────────────────────────────
T=0                  Episode 1 Starts
                     simulator.py:52
                     observations = env.reset()
                                    ↓
                     citylearn.py:649-650
                     for building in self.buildings:
                         building.reset()
                                    ↓
                     building.py:837
                     self.electrical_storage.reset()
                                    ↓
                     energy_model.py:603
                     self.__soc = [self.initial_soc]
                                                        SOC = [0.0]

T=0-1                Episode 1: Timestep Loop
                     simulator.py:55-78
                     while not env.done:
                       action = agent.select_actions()
                       next_obs, reward = env.step()
                       
                     env.step() calls:
                     building.step() → electrical_storage.charge(energy)
                                    ↓
                     energy_model.py:773
                     self.__soc.append(soc)
                                                        SOC = [0.0, 2.5]

T=1-2                Second timestep in Episode 1
                     agent decides charge/discharge
                     battery.charge() called again
                     self.__soc.append(soc)
                                                        SOC = [0.0, 2.5, 4.1]

...

T=8759              Last timestep of Episode 1
                     battery.charge() called
                     self.__soc.append(soc)
                                                        SOC = [0.0, 2.5, 4.1, 
                                                               3.7, 2.1, ...]

                     env.done = True
                     Episode 1 Loop Exits
                     Final Battery State: [0.0, 2.5, 4.1, ..., 3.7]
                     Current SOC (last value): 3.7 kWh
                                                        SOC = [0.0, ..., 3.7]
                                                        (8760 values total)
```

---

### **Episode 2: Reset Happens**

```
TIME                  EVENT                           BATTERY STATE
─────────────────────────────────────────────────────────────────────────
T=0                 Episode 1 ended
                    Previous Episode 1 final SOC: 3.7 kWh
                    (This value is NOT used in Episode 2)
                                                        SOC = [0.0, ..., 3.7]
                                                        (8760 values - HISTORY)

T=0.001              Episode 2 Starts
                     simulator.py:48
                     for episode in range(episodes):  # episode = 1 (2nd episode)
                     
T=0.002              simulator.py:52
                     observations = env.reset()  ◄─── RESET CALLED
                                    ↓
                     
                     citylearn.py:638-655
                     def reset(self):
                         for building in self.buildings:
                             building.reset()
                                    ↓
                     
                     building.py:829-842
                     def reset(self):
                         self.electrical_storage.reset()
                                    ↓
                     
                     energy_model.py:600-605
                     def reset(self):
                         super().reset()
                         self.__soc = [self.initial_soc]  ◄─── CRITICAL!
                         self.__energy_balance = [0.0]
                         
                     ✅ OLD SOC LIST DISCARDED
                     ✅ NEW SOC LIST CREATED
                                                        SOC = [0.0]
                                                        (Fresh Start!)

T=0-1                Episode 2: Timestep Loop
                     Agent decisions might be different
                     battery.charge() called
                     self.__soc.append(soc)
                                                        SOC = [0.0, 1.2]
                                                        (Different value than E1!)

T=1-2                Episode 2: Second timestep
                     battery.charge() called
                                                        SOC = [0.0, 1.2, 3.0]

...

T=8759               Episode 2: Last timestep
                     battery.charge() called
                                                        SOC = [0.0, 1.2, 3.0,
                                                               ..., 2.1]

                     env.done = True
                     Episode 2 Loop Exits
                     Final Battery State: [0.0, 1.2, 3.0, ..., 2.1]
                     Current SOC (last value): 2.1 kWh
                                                        SOC = [0.0, ..., 2.1]
                                                        (Different from E1!)
```

---

### **Episode 3, 4, ... 30: Pattern Repeats**

```
Each Episode Follows Same Pattern:

┌─────────────────────────────────────────────────┐
│         EPISODE N START                         │
│  observations = env.reset()                     │
│         ↓                                       │
│  self.__soc = [0.0]  ◄─── RESET AGAIN         │
└─────────────────────────────────────────────────┘
              ↓
  Episode N Execution (8760 timesteps)
  Battery evolves: [0.0] → [0.0, ...] → [..., FINAL_SOC_N]
              ↓
┌─────────────────────────────────────────────────┐
│         EPISODE N+1 START                       │
│  observations = env.reset()                     │
│         ↓                                       │
│  self.__soc = [0.0]  ◄─── RESET AGAIN         │
└─────────────────────────────────────────────────┘

RESULT: Each episode starts fresh at SOC=0.0 kWh
        Previous episode's final SOC is discarded
```

---

## 🔗 Complete Call Chain with Line Numbers

```
SIMULATOR ENTRY
↓
simulate.py:48
for episode in trange(env.schema['episodes']):
↓
simulate.py:52
observations = env.reset()
│
└──→ citylearn.py:638
    def reset(self):
    ├── Line 647: super().reset()
    ├── Line 649-650:
    │   for building in self.buildings:
    │       building.reset()
    │   ├── building.py:829
    │   │   def reset(self):
    │   │   ├── Line 833: super().reset()
    │   │   ├── Line 835: self.cooling_storage.reset()
    │   │   ├── Line 836: self.heating_storage.reset()
    │   │   ├── Line 838: self.dhw_storage.reset()
    │   │   └── Line 837: self.electrical_storage.reset()  ◄─ BATTERY HERE
    │   │       │
    │   │       └── energy_model.py:600
    │   │           def reset(self):
    │   │           ├── Line 603: self.__soc = [self.initial_soc]  ◄─ RESET!
    │   │           └── Line 604: self.__energy_balance = [0.0]
    │   │
    │   └── Returns to building.py:841-849
    │       (Reset tracking variables)
    │
    └── Returns to citylearn.py:652-654
        (Reset reward tracking)
        └── Line 655: return self.observations

↓
simulate.py:55-78
while not env.done:
    actions = agents.select_actions(observations)
    next_obs, reward, _, _ = env.step(actions)
    (Battery SOC evolves here)
    
↓
(Episode ends when env.done = True)

↓
simulate.py:48 (Next iteration)
for episode in range(episodes):  # episode increments
observations = env.reset()  ◄─ RESET CALLED AGAIN
(Repeat entire cycle)
```

---

## 📋 File Tracking Summary Table

| File | Line(s) | Method | Purpose | Evidence |
|------|---------|--------|---------|----------|
| **simulate.py** | 48-52 | `simulate()` | Episode loop, calls `env.reset()` | Entry point for reset |
| **simulate.py** | 55-78 | `simulate()` | Timestep loop, battery evolves | Battery SOC changes here |
| **citylearn.py** | 638-655 | `reset()` | Orchestrates environment reset | Calls `building.reset()` on all buildings |
| **building.py** | 829-842 | `reset()` | Resets all building components | Calls `electrical_storage.reset()` |
| **energy_model.py** | 600-605 | `reset()` | Resets storage state | **`self.__soc = [initial_soc]`** |
| **energy_model.py** | 760-773 | `charge()` | Evolves battery during episode | `self.__soc.append(soc)` |
| **energy_model.py** | 440-457 | `__init__()` | Initializes StorageDevice | Sets `initial_soc = 0.0` (default) |
| **settings.json** | - | Configuration | Defines episodes count | `"train_episodes": 1` (configurable) |

---

## 🎯 Key Checkpoints (Verification Points)

To verify battery reset at each episode, check these values:

### **Before Reset (End of Episode N):**
```python
battery_soc_before = env.buildings[0].electrical_storage.soc
# Example: [0.0, 2.5, 4.1, 3.7, ..., 2.1]  ← 8760 values
print(f"SOC list length before reset: {len(battery_soc_before)}")  # 8760
print(f"Last SOC before reset: {battery_soc_before[-1]}")  # 2.1
```

### **After Reset (Beginning of Episode N+1):**
```python
# After env.reset() is called
battery_soc_after = env.buildings[0].electrical_storage.soc
# Example: [0.0]  ← Single initial value
print(f"SOC list length after reset: {len(battery_soc_after)}")  # 1
print(f"First SOC after reset: {battery_soc_after[0]}")  # 0.0
```

### **Verification Script:**
```python
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(schema)
agents = env.load_agent()

for episode in range(3):  # 3 episodes
    print(f"\n{'='*50}")
    print(f"EPISODE {episode}")
    print(f"{'='*50}")
    
    # Before reset
    if episode > 0:
        print(f"Before reset - SOC length: {len(env.buildings[0].electrical_storage.soc)}")
        print(f"Before reset - Last SOC: {env.buildings[0].electrical_storage.soc[-1]:.2f}")
    
    # Reset
    observations = env.reset()
    print(f"After reset - SOC: {env.buildings[0].electrical_storage.soc}")
    
    # Execute episode
    while not env.done:
        actions = agents.select_actions(observations)
        next_obs, reward, _, _ = env.step(actions)
        observations = next_obs
    
    print(f"Episode {episode} complete - Final SOC: {env.buildings[0].electrical_storage.soc[-1]:.2f}")
```

---

## 📊 Visual State Diagram

```
BATTERY STATE ACROSS 3 EPISODES:

EPISODE 1:
┌─ Reset called ────────┐
│ SOC = [0.0]           │
│ Timesteps loop        │
│ SOC evolves:          │
│ [0.0, 2.5, 4.1,       │
│  3.7, ..., 2.1]       │
│ Length = 8760         │
└───────────────────────┘
         ↓
    Final value: 2.1 kWh
    (DISCARDED for next episode)
         ↓
EPISODE 2:
┌─ Reset called ────────┐
│ SOC = [0.0]           │  ◄─── Fresh start, not 2.1!
│ Timesteps loop        │
│ SOC evolves:          │
│ [0.0, 1.2, 3.0,       │
│  2.5, ..., 1.8]       │
│ Length = 8760         │
└───────────────────────┘
         ↓
    Final value: 1.8 kWh
    (DISCARDED for next episode)
         ↓
EPISODE 3:
┌─ Reset called ────────┐
│ SOC = [0.0]           │  ◄─── Fresh start, not 1.8!
│ Timesteps loop        │
│ SOC evolves:          │
│ [0.0, 3.2, 2.1,       │
│  4.5, ..., 3.1]       │
│ Length = 8760         │
└───────────────────────┘
         ↓
    Final value: 3.1 kWh

KEY INSIGHT:
Each episode STARTS fresh at 0.0 kWh
Previous episode's final value is NOT used
```

---

## ✅ Conclusion with Complete Evidence

**Battery Reset Flow Summary:**

1. **Initiator**: `simulator.py:52` calls `env.reset()`
2. **Orchestrator**: `citylearn.py:638-650` coordinates the reset
3. **Implementer**: `building.py:829-842` calls component resets
4. **Executor**: `energy_model.py:600-605` performs actual reset: `self.__soc = [initial_soc]`
5. **Result**: Battery starts at 0.0 kWh for EVERY episode

**This happens for ALL 30 episodes** (or however many configured in settings.json)

**No state carries over between episodes** - Each episode is independent

---

## 📚 Related Code Sections

### SOC Storage Implementation
```python
# energy_model.py : Line 515-520
@property
def soc(self) -> List[float]:
    r"""State of charge time series in [kWh]."""
    return self.__soc  # List grows during episode, reset at episode start
```

### Energy Balance
```python
# energy_model.py : Line 525-530  
@property
def energy_balance(self) -> List[float]:
    r"""Charged/discharged energy time series in [kWh]."""
    return self.__energy_balance  # Also reset at episode start
```

### Initial SOC from Schema
```python
# From schema.json (example)
{
    "electrical_storage": {
        "capacity": 10.0,
        "nominal_power": 5.0,
        "initial_soc": 0.5  // Can be configured, default 0.0
    }
}
```

---

## 🔬 Testing the Reset Behavior

To verify this behavior in your project:

```bash
cd D:\Research\RL\Degradation analysis\201\src

# Add this to simulate.py for debugging:
# Before line 55, add:
print(f"Episode {episode}: Battery SOC before timestep loop = {env.buildings[0].electrical_storage.soc}")

# Run simulation
python -m simulate <schema> <simulation_id>

# Output should show:
# Episode 0: Battery SOC before timestep loop = [0.0]
# Episode 1: Battery SOC before timestep loop = [0.0]  # NOT [3.7]
# Episode 2: Battery SOC before timestep loop = [0.0]  # NOT [2.1]
```

