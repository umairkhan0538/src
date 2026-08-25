import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
from citylearn.data import DataSet
from citylearn.utilities import read_json, write_json

ROOT_DIRECTORY = os.path.join(*Path(os.path.dirname(__file__)).absolute().parts[0:-1])
DATA_DIRECTORY = r'D:\Research\Citylearn_related\Citylearn_data\data'

def get_settings():
    settings_path = os.path.join(Path(__file__).parent, 'settings.json')
    return read_json(settings_path)

def set_dataset():
    settings = get_settings()
    dataset_name = settings['dataset_name']
    destination = os.path.join(DATA_DIRECTORY, dataset_name)
    if os.path.isdir(destination):
        shutil.rmtree(destination)
    DataSet.copy(dataset_name, destination_directory=DATA_DIRECTORY)

def get_reward_configuration(settings):
    """Return the validated global NormCCEReward configuration."""
    selected_buildings = settings[
        'include_buildings']
    if len(selected_buildings) != 1:
        raise ValueError(
            'This experiment currently requires exactly one selected '
            'building per training run. Received: {}'.format(selected_buildings))

    building_name = 'Building_{}'.format(selected_buildings[0])
    reward_settings = settings['reward_functions']['NormCCEReward']

    if 'type' not in reward_settings:
        raise KeyError("NormCCEReward settings are missing the 'type' field.")

    if 'attributes' not in reward_settings:
        raise KeyError("NormCCEReward settings are missing the 'attributes' block.")

    # All buildings use the same global objective-specific references.
    reward_attributes = reward_settings['attributes'].copy()

    required_attributes = [
        'grid_weight',
        'cost_weight',
        'emission_weight',
        'avoidable_export_reference',
        'net_consumption_reference',
        'cost_reference',
        'emission_reference',
        'soc_min',
        'soc_max',
        'export_tariff',
        'terminal_soc_weight',
        'epsilon']

    missing_attributes = [attribute for attribute in required_attributes if attribute not in reward_attributes]

    if missing_attributes:
        old_reference_message = ''

        if 'energy_reference' in reward_attributes:
            old_reference_message = (
                " The obsolete 'energy_reference' field is still "
                "present. Replace it with "
                "'avoidable_export_reference'.")

        raise KeyError(
            'Missing NormCCEReward attributes: {}.{}'.format(missing_attributes, old_reference_message ))

    # Prevent accidental reuse of the old normalization design.
    if 'energy_reference' in reward_attributes:
        raise KeyError(
            "The obsolete 'energy_reference' field must be removed "
            "from the NormCCEReward attributes.")

    numerical_attributes = [
        'grid_weight',
        'cost_weight',
        'emission_weight',
        'avoidable_export_reference',
        'net_consumption_reference',
        'cost_reference',
        'emission_reference',
        'soc_min',
        'soc_max',
        'export_tariff',
        'terminal_soc_weight',
        'epsilon']

    for attribute in numerical_attributes:
        try:
            value = float(reward_attributes[attribute])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "NormCCEReward attribute '{}' must be numeric. "
                "Received: {!r}.".format(attribute,reward_attributes[attribute])) from error

        if not np.isfinite(value):
            raise ValueError(
                "NormCCEReward attribute '{}' must be finite. "
                "Received: {}.".format(attribute,value))

        reward_attributes[attribute] = value

    weights = np.asarray(
        [reward_attributes['grid_weight'],
         reward_attributes['cost_weight'],
         reward_attributes['emission_weight']],dtype=float)

    if np.any(weights < 0.0):
        raise ValueError('NormCCEReward weights must be non-negative.')

    if not np.isclose(weights.sum(),1.0, rtol=1e-6,atol=1e-6):
        raise ValueError(
            'NormCCEReward weights must sum to 1.0. '
            'Received sum: {:.10f}.'.format(
                weights.sum()))

    references = np.asarray(
        [
            reward_attributes['avoidable_export_reference'],
            reward_attributes['net_consumption_reference'],
            reward_attributes['cost_reference'],
            reward_attributes['emission_reference']],dtype=float)

    if np.any(references <= 0.0):
        raise ValueError(
            'All NormCCEReward normalization references '
            'must be greater than zero.')

    if not (0.0<= reward_attributes['soc_min']< reward_attributes['soc_max']<= 1.0):
        raise ValueError(
            'SOC limits must satisfy '
            '0 <= soc_min < soc_max <= 1.')

    # Optional parameters used by the revised reward class.
    reward_attributes.setdefault(
        'validation_tolerance',1e-6)
    reward_attributes.setdefault(
        'store_diagnostic_history',False)
    print('Using global NormCCEReward parameters for {}:'.format(building_name ))
    print(' Grid weight:',reward_attributes['grid_weight'])
    print(' Cost weight:',reward_attributes['cost_weight'])
    print(' Emission weight:',reward_attributes['emission_weight'])
    print(' Avoidable-export reference:',reward_attributes['avoidable_export_reference'])
    print(' Net-consumption reference:',reward_attributes[ 'net_consumption_reference'])
    print(' Cost reference:',reward_attributes['cost_reference'])
    print(' Emission reference:',reward_attributes['emission_reference'])
    print(' SOC range: [{}, {}]'.format(reward_attributes['soc_min'],reward_attributes['soc_max']))
    print(' Export tariff:',reward_attributes['export_tariff'])
    print('  Terminal SOC weight:',reward_attributes['terminal_soc_weight'])
    print(' Store diagnostic history:',reward_attributes['store_diagnostic_history'])
    return {'type': reward_settings['type'],'attributes': reward_attributes}

  
def schema_update(experiment, **kwargs):
    settings = get_settings()
    dataset_path = os.path.join(DATA_DIRECTORY, settings['dataset_name'])
    schema = read_json(os.path.join(dataset_path, 'schema.json'))

    for observation_name in schema['observations']:
        schema['observations'][observation_name]['active'] = (
            observation_name in settings['observations'])
    for building_name in schema['buildings']:
        building_id = int(building_name.split('_')[-1])
        schema['buildings'][building_name]['include'] = (
            building_id in settings['include_buildings'])
    schema['central_agent'] = settings['central_agent']
    kwargs['schema'] = schema
    experiment_function = get_experiment_function(experiment)
    experiment_function(experiment, **kwargs)

def reference_rbc(experiment, **kwargs):
    settings = get_settings()
    schema = kwargs['schema']
    selected_building = settings['include_buildings'][0]

    schema.update({
        'simulation_start_time_step': settings['train_start_time_step'],
        'simulation_end_time_step': settings['test_end_time_step'],
        'episodes': 1,
        'root_directory': os.path.join(DATA_DIRECTORY, settings['dataset_name']),
        'agent': {
            'type': settings['rbcagent']['type'],
            'attributes': schema['agent']['attributes']}})

    suffix = f'schema_building_{selected_building}_energy_report.json'
    write_and_run_script(experiment, schema, suffix=suffix)


def sac(experiment, **kwargs):
    settings = get_settings()
    schema = kwargs['schema']
    selected_building = settings['include_buildings'][0]
    reward_configuration = get_reward_configuration(settings)

    agent_attributes = settings['drlagent']['attributes'].copy()
    schema.update({
        'simulation_start_time_step': settings['train_start_time_step'],
        'simulation_end_time_step': settings['test_end_time_step'],
        'episodes': settings['train_episodes'],
        'root_directory': os.path.join(DATA_DIRECTORY, settings['dataset_name']),
        'reward_function': reward_configuration,
        'agent': {
            'type': settings['drlagent']['type'],
            'attributes': agent_attributes}})

    schema['agent']['attributes']['deterministic_start_time_step'] = (settings['train_episodes'] - 1) * 8760
    suffix = f'schema_building_{selected_building}_updated.json'
    extra_args = '--save_episode_agent 10 --deterministic'
    write_and_run_script(experiment, schema, suffix=suffix, extra_args=extra_args)


def write_and_run_script(experiment, schema, suffix, extra_args=''):
    shell_directory = os.path.join(ROOT_DIRECTORY, 'misc', 'shell')
    schema_directory = os.path.join(ROOT_DIRECTORY, 'misc', 'exp_schema')

    os.makedirs(shell_directory, exist_ok=True)
    os.makedirs(schema_directory, exist_ok=True)

    schema_file = os.path.join(schema_directory, f'{experiment}_{suffix}')
    shell_file = os.path.join(shell_directory, f'{experiment}GCPVB.sh')
    simulation_file = os.path.join(ROOT_DIRECTORY, 'src', 'simulate.py')

    write_json(schema_file, schema)

    command = (f'python "{simulation_file}" "{schema_file}" 'f'{experiment} {extra_args}').strip()

    with open(shell_file, 'w') as file:
        file.write(command + '\n')

    print(f'Schema written to: {schema_file}')
    print(f'Command written to: {shell_file}')
    print(f'Command: {command}')


def run(experiment, **kwargs):
    shell_file = os.path.join(ROOT_DIRECTORY, 'misc', 'shell',f'{experiment}GCPVB.sh')

    with open(shell_file, 'r') as file:
        commands = file.read().strip().split('\n')

    print(f'Using {cpu_count()} CPU workers to run {len(commands)} job(s)...')

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(subprocess.run, args=command, shell=True) for command in commands]

        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as error:
                print(f'Error: {error}')


def get_experiment_function(experiment):
    return {'reference_rbc': reference_rbc, 'sac': sac }[experiment]

def get_experiments():
    return ['reference_rbc', 'sac']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=get_experiments(), type=str)

    subparsers = parser.add_subparsers(dest='subcommands', required=True)

    parser_schema = subparsers.add_parser('schema_update')
    parser_schema.set_defaults(func=schema_update)

    parser_run = subparsers.add_parser('run')
    parser_run.set_defaults(func=run)

    args = parser.parse_args()
    kwargs = {
        key: value for key, value in vars(args).items()
        if key not in ['func', 'subcommands'] }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())