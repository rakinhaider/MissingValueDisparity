import os
from folktables import ACSDataSource, ACSPublicCoverage
from datasets.realworld.realworld_base import RealWorldBase


def _income_quantizer(x):
    if x >= 50000:
        return 1
    else:
        return 0


def _age_quantizer(x):
    if x <= 45:
        return 1
    else:
        return 0


def _race_privilge_mapper(x):
    if x == 1: return 1
    else: return 0


class FolkIncomeDataset(RealWorldBase):
    def __init__(self, label_names=['PINCP'],
                 protected_attribute_names=['AGEP'], favorable_label=1,
                 year='2018', horizon='1-Year', survey='person',
                 force_download=False,
                 ):
        root_dir = 'data/raw/folktables'
        data_source = ACSDataSource(survey_year=year, horizon=horizon,
                                    survey=survey,
                                    root_dir=root_dir)
        to_download = True
        if os.path.exists(os.path.join(root_dir, year, horizon)):
            to_download = False or force_download
        acs_data = data_source.get_data(states=["CA"], download=to_download)
        columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP',
                   'RELP', 'WKHP', 'SEX', 'RAC1P', 'PINCP']
        df = acs_data[columns]
        df = df.dropna()

        df['AGEP'] = df['AGEP'].astype(int).apply(_age_quantizer)
        df['PINCP'] = df['PINCP'].astype(int).apply(_income_quantizer)
        for s in ['RAC1P', 'SEX', 'AGEP']:
            if s not in protected_attribute_names:
                df = df.drop(columns=[s])

        # print(df['AGEP'].value_counts())
        # print(df['PINCP'].value_counts())
        # print(df)
        super(FolkIncomeDataset, self).__init__(
            df=df, label_names=label_names,
            protected_attribute_names=protected_attribute_names,
            favorable_label=favorable_label,
            unfavorable_label=1 - favorable_label)


class FolkPubCoverageDataset(RealWorldBase):
    def __init__(self, label_names=['PUBCOV'],
                 protected_attribute_names=['RAC1P'], favorable_label=1,
                 year='2018', horizon='1-Year', survey='person',
                 force_download=False,
                 ):
        root_dir = 'data/raw/folktables'
        data_source = ACSDataSource(survey_year=year, horizon=horizon,
                                    survey=survey,
                                    root_dir=root_dir)
        to_download = True
        if os.path.exists(os.path.join(root_dir, year, horizon)):
            to_download = False or force_download
        acs_data = data_source.get_data(states=["CA"], download=to_download)
        df, labels, groups = ACSPublicCoverage.df_to_pandas(acs_data)
        df[labels.columns[0]] = labels
        df[groups.columns[0]] = groups

        df['AGEP'] = df['AGEP'].astype(int).apply(_age_quantizer)
        df['PINCP'] = df['PINCP'].astype(int).apply(_income_quantizer)
        df['RAC1P'] = df['RAC1P'].astype(int).apply(_race_privilge_mapper)
        df['PUBCOV'] = df['PUBCOV'].astype(int)
        print(df)
        for s in ['RAC1P', 'SEX', 'AGEP']:
            if s not in protected_attribute_names:
                df = df.drop(columns=[s])

        super(FolkPubCoverageDataset, self).__init__(
            df=df, label_names=label_names,
            protected_attribute_names=protected_attribute_names,
            favorable_label=favorable_label,
            unfavorable_label=1 - favorable_label)
