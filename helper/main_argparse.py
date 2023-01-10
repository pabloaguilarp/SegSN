import argparse
import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--dataset-lf', '-d',
        type=str,
        required=True,
        help='Folder where scans in LiDAR frame are',
    )
    parser.add_argument(
        '--dataset-wf', '-w',
        type=str,
        required=False,
        help='Folder where scans in world frame are',
    )
    parser.add_argument(
        '--output-folder', '-o',
        type=str,
        required=False,
        help='Folder where predictions will be saved'
    )
    parser.add_argument(
        '--save-ranges', '-r',
        type=str,
        required=False,
        help='Save also .range files'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )
    parser.add_argument(
        '--monte-carlo', '-c',
        type=int, default=30,
        help='Number of samplings per scan'
    )
    parser.add_argument(
        '--use-visualizer', '-v',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Use visualizer'
    )
    parser.add_argument(
        '--filter-labels', '-f',
        nargs='+',
        default=None,
        help='Filter labels in visualization'
    )
    parser.add_argument(
        '--use-semantics', '-l',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Use semantics'
    )
    parser.add_argument(
        '--data-source', '-s',
        type=str, nargs='?',
        const=True, default='kitti',
        help='Data source [kitti | custom]'
    )
    parser.add_argument(
        '--darkening-factor', '-df',
        type=float, nargs='?',
        const=True, default='1.0',
        help='Darkening factor'
    )
    parser.add_argument(
        '--compression-threshold', '-ct',
        type=float, nargs='?',
        const=True, default='50.0',
        help='Compression threshold'
    )
    parser.add_argument(
        '--compression-ratio', '-cr',
        type=float, nargs='?',
        const=True, default='0.1',
        help='Compression ratio'
    )

    flags, unparsed = parser.parse_known_args()
    return flags, unparsed


def load_pretrained(model: str):
    # open arch config file
    arch = None
    data = None
    try:
        print("Opening arch config file from %s" % model)
        arch = yaml.safe_load(open(model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % model)
        data = yaml.safe_load(open(model + "/data_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    return arch, data
