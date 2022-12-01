import wget
import os
import argparse
import zipfile
import tempfile
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aif360-folder',
                        help='The folder where aif360 packege is installed.',
                        default="myvenv/Lib/site-packages/aif360/")
    parser.add_argument('--dataset', '-d', default='compas')

    file_name = ""
    args = parser.parse_args()

    if args.dataset == 'compas':
        out = os.path.join(args.aif360_folder, 'data', 'raw', 'compas',
                           'compas-scores-two-years.csv')
        wget.download('https://raw.githubusercontent.com/propublica/'
                  'compas-analysis/master/compas-scores-two-years.csv',
                  out)

    elif args.dataset == 'bank':
        tempdir = tempfile.mkdtemp()
        target = os.path.join(tempdir, 'bank-additional.zip')
        wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases'
                      '/00222/bank-additional.zip', target)

        with zipfile.ZipFile(target, 'r') as zip_ref:
            zip_ref.extractall(tempdir)

        destination = os.path.join(args.aif360_folder, 'data', 'raw', 'bank')
        for file in ['bank-additional.csv', 'bank-additional-full.csv']:
            shutil.copy(os.path.join(tempdir, 'bank-additional', file),
                        os.path.join(destination, file))

    elif args.dataset == 'german':
        dir = os.path.join(args.aif360_folder, 'data', 'raw', 'german')
        baseurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' \
                  'statlog/german/'

        for file in ['german.data', 'german.doc']:
            target = os.path.join(dir, file)
            wget.download(baseurl + file, target)
