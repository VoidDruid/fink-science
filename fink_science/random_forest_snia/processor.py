# Copyright 2019-2021 AstroLab Software
# Author: Julien Peloton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType, StringType

import pandas as pd
import numpy as np

import os

from fink_science.conversion import mag2fluxcal_snana

from fink_science import __file__
from fink_science.utilities import load_scikit_model
from fink_science.random_forest_snia.classifier_sigmoid import get_sigmoid_features_dev
from fink_science.random_forest_snia.classifier_sigmoid import RF_FEATURE_NAMES
from fink_science.random_forest_snia.classifier_sigmoid import return_list_of_sn_host

from fink_science.tester import spark_unit_tests

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_sigmoid_full(jd, fid, magpsf, sigmapsf, cdsxmatch, ndethist, model=None) -> pd.Series:
    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier (sigmoid fit).

    You need to run the SIMBAD crossmatch before.

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (vectors of floats)
    fid: Spark DataFrame Column
        Filter IDs (vectors of ints)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error (vectors of floats)
    cdsxmatch: Spark DataFrame Column
        Type of object found in Simbad (string)
    ndethist: Spark DataFrame Column
        Column containing the number of detection by ZTF at 3 sigma (int)
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/default-model.obj` is loaded.

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    ----------
    >>> from fink_science.xmatch.processor import cdsxmatch
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    >>> colnames = [df['objectId'], df['candidate.ra'], df['candidate.dec']]
    >>> df = df.withColumn('cdsxmatch', cdsxmatch(*colnames))

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> df = df.withColumn('pIa', rfscore_sigmoid_full(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    5

    >>> df.filter(df['pIa'] > 0.5).select(['rf_snia_vs_nonia', 'pIa']).show()
    +----------------+-----+
    |rf_snia_vs_nonia|  pIa|
    +----------------+-----+
    |           0.839|0.597|
    |           0.887|0.629|
    |           0.785|0.596|
    |            0.88|0.641|
    |           0.777|0.611|
    +----------------+-----+
    <BLANKLINE>

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> args += [F.lit(model_path_sigmoid)]
    >>> df = df.withColumn('pIa', rfscore_sigmoid_full(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    5

    >>> df.agg({"pIa": "max"}).collect()[0][0] < 1.0
    True
    """
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 3

    mask *= (ndethist.astype(int) <= 20)

    list_of_sn_host = return_list_of_sn_host()
    mask *= cdsxmatch.apply(lambda x: x in list_of_sn_host)

    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    # add an exploded column with SNID
    df_tmp = pd.DataFrame.from_dict(
        {
            'jd': jd[mask],
            'SNID': range(len(jd[mask]))
        }
    )
    df_tmp = df_tmp.explode('jd')

    # compute flux and flux error
    data = [mag2fluxcal_snana(*args) for args in zip(
        magpsf[mask].explode(),
        sigmapsf[mask].explode())]
    flux, error = np.transpose(data)

    # make a Pandas DataFrame with exploded series
    pdf = pd.DataFrame.from_dict({
        'SNID': df_tmp['SNID'],
        'MJD': df_tmp['jd'],
        'FLUXCAL': flux,
        'FLUXCALERR': error,
        'FLT': fid[mask].explode().replace({1: 'g', 2: 'r'})
    })

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/default-model_sigmoid.obj'
        clf = load_scikit_model(model)

    test_features = []
    flag = []
    for id in np.unique(pdf['SNID']):
        pdf_sub = pdf[pdf['SNID'] == id]
        features = get_sigmoid_features_dev(pdf_sub)
        if (features[0] == 0) or (features[6] == 0):
            flag.append(False)
        else:
            flag.append(True)
        test_features.append(features)

    flag = np.array(flag, dtype=np.bool)

    # Make predictions
    probabilities = clf.predict_proba(test_features)
    probabilities[~flag] = 0.0

    # Take only probabilities to be Ia
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities.T[1]

    return pd.Series(to_return)

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def extract_features_rf_snia(jd, fid, magpsf, sigmapsf) -> pd.Series:
    """ Return the features used by the RF classifier.

    There are 12 features. Order is:
    a_g,b_g,c_g,snratio_g,chisq_g,nrise_g,
    a_r,b_r,c_r,snratio_r,chisq_r,nrise_r

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error

    Returns
    ----------
    features: list of str
        List of string.

    Examples
    ----------
    >>> from pyspark.sql.functions import split
    >>> from pyspark.sql.types import FloatType
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> df = df.withColumn('features', extract_features_rf_snia(*args))

    >>> for name in RF_FEATURE_NAMES:
    ...   index = RF_FEATURE_NAMES.index(name)
    ...   df = df.withColumn(name, split(df['features'], ',')[index].astype(FloatType()))

    # Trigger something
    >>> df.agg({RF_FEATURE_NAMES[0]: "min"}).collect()[0][0]
    -84236.9296875
    """
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 3
    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    # add an exploded column with SNID
    df_tmp = pd.DataFrame.from_dict(
        {
            'jd': jd[mask],
            'SNID': range(len(jd[mask]))
        }
    )
    df_tmp = df_tmp.explode('jd')

    # compute flux and flux error
    data = [mag2fluxcal_snana(*args) for args in zip(
        magpsf[mask].explode(),
        sigmapsf[mask].explode())]
    flux, error = np.transpose(data)

    # make a Pandas DataFrame with exploded series
    pdf = pd.DataFrame.from_dict({
        'SNID': df_tmp['SNID'],
        'MJD': df_tmp['jd'],
        'FLUXCAL': flux,
        'FLUXCALERR': error,
        'FLT': fid[mask].explode().replace({1: 'g', 2: 'r'})
    })

    test_features = []
    for id in np.unique(pdf['SNID']):
        pdf_sub = pdf[pdf['SNID'] == id]
        features = get_sigmoid_features_dev(pdf_sub)
        test_features.append(features)

    to_return_features = np.zeros((len(jd), len(RF_FEATURE_NAMES)), dtype=float)
    to_return_features[mask] = test_features

    concatenated_features = [
        ','.join(np.array(i, dtype=str)) for i in to_return_features
    ]

    return pd.Series(concatenated_features)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    model_path_sigmoid = '{}/data/models/default-model_sigmoid.obj'.format(path)
    globs["model_path_sigmoid"] = model_path_sigmoid

    # Run the test suite
    spark_unit_tests(globs)
