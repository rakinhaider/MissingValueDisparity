import warnings

warnings.filterwarnings("ignore")
import os
import logging
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import get_standard_dataset


if __name__ == '__main__':
    for dataset in ['compas', 'adult', 'bank', 'german', 'pima', 'heart',
                    'folkincome']:
        data = get_standard_dataset(dataset)
        print(f'Name of the dataset: {dataset}')
        print(f'\tNumber of samples: {data.features.shape[0]}')
        print(f'\tNumber of features {len(data.feature_names)}')
