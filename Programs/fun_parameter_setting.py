from __future__ import print_function
import argparse
import datetime
import copy

def parameter_setting(params, paths, json_name, code_cloud):

    parser = argparse.ArgumentParser(description="Time Series Anomaly Detection with Entropic/Information measures")

    # Esperiment name
    parser.add_argument('--exp-name', type=str, default=params.get('exp_name', json_name[:json_name.find('.json')]) , help='experiment name')

    # Dataset
    parser.add_argument('--dataset', type=str, default=params.get('dataset', 'EEG'), help='select the dataset (synthetic or EEG)')

    # Simulation general settings
    parser.add_argument('--deltaX', nargs='+', type=int, default=params.get('deltaX', 2**9), help='deltaX values')
    parser.add_argument('--x-discr', nargs='+', type=int, default=params.get('x_discr', 2048), help='x discretization points')

    # Probability distribution transformation
    parser.add_argument('--prob_distr_type', type=str, default=params.get('prob_distr_type', 'KDE'), help='Probability distribution type')

    # Specific transformation settings
    parser.add_argument('--h-opt-strategy', type=str, default=params.get('h_opt_strategy', 'plug_in'), help='Optimization strategy')
    if params['h_opt_strategy'] == 'cost':
        parser.add_argument('--h-Delta-list', type=list, default=params['h_Delta_list'], help='h_Delta_list')
    else:
        if "JSD" in params['h_opt_strategy']:
            parser.add_argument('--h-plato-th', nargs='+', type=float, default=params.get('h_plato_th', 0.02), help='h plato JSD threshold')
        parser.add_argument('--DeltaX-grid', type=list, default=params.get('DeltaX_grid', [4, 13]), help='DeltaX grid search range')
        parser.add_argument('--grid-exp-base', type=float, default=params.get('grid_exp_base', 2.0), help='grid exp base')            

    # synthetic settings
    if params['dataset'] == 'synthetic':
        parser.add_argument('--sampling-rate', type=list, default=params.get('sampling_rate', 4096), help='Sampling rate')
        parser.add_argument('--amps', type=list, default=params.get('amps', [1.0]), help='List of normal amplitudes')
        parser.add_argument('--freqs', type=list, default=params.get('freqs', [440.0]), help='List of normal frequencies')
        parser.add_argument('--amps-anom', type=list, default=params.get('amps_anom', [1.0]), help='List of normal amplitudes')
        parser.add_argument('--freqs-anom', type=list, default=params.get('freqs_anom', [440.0]), help='List of normal frequencies')
        parser.add_argument('--t-total', type=list, default=params.get('t_total', 10.0), help='total simulation duration')
        parser.add_argument('--t0-anomaly', type=list, default=params.get('t0_anomaly', 5.0), help='t start anomaly')
        parser.add_argument('--anomaly-type', type=str, default=params.get('anomaly_type', 'linear'), help='anomaly_type')
    
        parser.add_argument('--gaussian-noise-mean', type=list, default=params.get('gaussian_noise_mean', [0.0]), help='List of normal amplitudes')
        parser.add_argument('--gaussian-noise-std', type=list, default=params.get('gaussian_noise_std', [0.0]), help='List of normal frequencies')
        parser.add_argument('--gaussian-noise-mean-anom', type=list, default=params.get('gaussian_noise_mean_anom', [0.0]), help='List of anomaly amplitudes')
        parser.add_argument('--gaussian-noise-std-anom', type=list, default=params.get('gaussian_noise_std_anom', [0.0]), help='List of anomaly frequencies')

    # EEG settings
    elif params['dataset'] == 'EEG':
        parser.add_argument('--chb_indx', nargs='+', type=int, default=params.get('chb_indx', 1), help='Patient indices')
        parser.add_argument('--chb_path', nargs='+', type=str, default=params['chb_path'])
        parser.add_argument('--min_Pre_Ict', type=int, default=params.get('min_Pre_Ict', 3), help='Minutes before Ictal')
        parser.add_argument('--min_Post_Ict', type=int, default=params.get('min_Post_Ict', 3), help='Minutes after Ictal')
        
    parser.add_argument('--IM-list', type=list, default=params.get('IM_list', ["S"]), help='List of IM to compute against time')

    #args = parser.parse_args()
    args = copy.deepcopy(parser.parse_args())

    args.code_cloud = code_cloud
        
    now = datetime.datetime.now()
    args.model_name = f'{args.dataset}_{now.strftime("%m_%d_%H_%M_%S")}'

    for key, value in paths.items():
        setattr(args, key, value)

    print("Model Settings:")
    for key, value in vars(args).items():
        print(f"{key} --> {value}")

    return args