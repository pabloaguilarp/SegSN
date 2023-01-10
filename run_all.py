import argparse
import os


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--folder-lf', '-l',
        type=str,
        required=True,
        help='Folder where scans in LiDAR frame are',
    )
    parser.add_argument(
        '--folder-wf', '-w',
        type=str,
        required=True,
        help='Folder where scans in world frame are',
    )
    parser.add_argument(
        '--folder-results', '-r',
        type=str,
        required=True,
        help='Folder where results will be saved',
    )
    parser.add_argument(
        '--command', '-cmd',
        type=str,
        required=True,
        help='Python execution command',
    )
    flags, unparsed = parser.parse_known_args()
    return flags, unparsed


def get_folders_in(folder: str):
    ret = []
    for name in os.listdir(folder):
        ret.append(name)
    return ret


if __name__ == '__main__':
    flags, unparsed = parse_args(argparse.ArgumentParser("./run_all.py"))
    folder_lf = flags.folder_lf
    folder_wf = flags.folder_wf
    folder_results = flags.folder_results

    folders_lf = get_folders_in(folder=folder_lf)
    folders_wf = get_folders_in(folder=folder_wf)

    print(f"Folders in LF: {len(folders_lf)}")
    print(f"Folders in WF: {len(folders_wf)}")

    for index in range(len(folders_lf)):
        print(f"LF: {folders_lf[index]}\tWF: {folders_wf[index]}")

        output_folder = os.path.join(folder_results, folders_wf[index])
        output_folder_pred = os.path.join(output_folder, "predictions")
        output_folder_rng = os.path.join(output_folder, "ranges")

        os.mkdir(output_folder)
        os.mkdir(output_folder_pred)
        os.mkdir(output_folder_rng)

        args: str = str(" -d \"" + os.path.join(folder_lf, folders_lf[index]) + "\"") + \
                    str(" -w \"" + os.path.join(folder_wf, folders_wf[index]) + "\"") + \
                    str(" -o \"" + output_folder_pred + "\"") + \
                    str(" -r \"" + output_folder_rng + "\"") + \
                    str(" -m ./pretrained -s custom")

        cmd = flags.command + args
        os.system(cmd)
