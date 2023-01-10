import argparse
from helper import main_argparse as helper
from salsanext_utils.user import User

if __name__ == '__main__':
    print('Starting inferrer')
    flags, unparsed = helper.parse_args(argparse.ArgumentParser("./main.py"))
    arch, data = helper.load_pretrained(flags.model)

    user = User(flags, arch, data)
    user.infer_dataset()
