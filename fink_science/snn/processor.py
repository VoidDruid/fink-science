# Copyright 2020-2022 AstroLab Software
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
from pyspark.sql.types import DoubleType

from supernnova.validation.validate_onthefly import classify_lcs

import pandas as pd
import numpy as np

import os

from fink_science import __file__
from fink_science.conversion import mag2fluxcal_snana
from fink_science.snn.utilities import reformat_to_df
from fink_science.snn.utilities import return_list_of_sn_host

from fink_science.tester import spark_unit_tests

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def snn_ia(candid, jd, fid, magpsf, sigmapsf, roid, cdsxmatch, jdstarthist, model_name, model_ext=None) -> pd.Series:
    """ Compute probabilities of alerts to be SN Ia using SuperNNova

    Parameters
    ----------
    candid: Spark DataFrame Column
        Candidate IDs (int64)
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    model_name: Spark DataFrame Column
        SuperNNova pre-trained model. Currently available:
            * snn_snia_vs_nonia
            * snn_sn_vs_all
    model_ext: Spark DataFrame Column, optional
        Path to the trained model (overwrite `model`). Default is None

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    ----------
    >>> from fink_science.xmatch.processor import cdsxmatch
    >>> from fink_science.asteroids.processor import roid_catcher
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Add SIMBAD field
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

    # Add SSO field
    >>> args_roid = [
    ...    'cjd', 'cmagpsf',
    ...    'candidate.ndethist', 'candidate.sgscore1',
    ...    'candidate.ssdistnr', 'candidate.distpsnr1']
    >>> df = df.withColumn('roid', roid_catcher(*args_roid))

    # Perform the fit + classification (default model)
    >>> args = ['candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf']
    >>> args += [F.col('roid'), F.col('cdsxmatch'), F.col('candidate.jdstarthist')]
    >>> args += [F.lit('snn_snia_vs_nonia')]
    >>> df = df.withColumn('pIa', snn_ia(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    7

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in ['candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf']]
    >>> args += [F.col('roid'), F.col('cdsxmatch'), F.col('candidate.jdstarthist')]
    >>> args += [F.lit(''), F.lit(model_path)]
    >>> df = df.withColumn('pIa', snn_ia(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    7
    """
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 1

    mask *= jd.apply(lambda x: float(x[-1])) - jdstarthist.astype(float) <= 90

    mask *= roid.astype(int) != 3

    list_of_sn_host = return_list_of_sn_host()
    mask *= cdsxmatch.apply(lambda x: x in list_of_sn_host)

    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    if model_ext is not None:
        # take the first element of the Series
        model = model_ext.values[0]
    else:
        # Load pre-trained model
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/snn_models/{}/model.pt'.format(model_name.values[0])

    # add an exploded column with SNID
    df_tmp = pd.DataFrame.from_dict(
        {
            'jd': jd[mask],
            'SNID': [str(i) for i in candid[mask].values]
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

    # Compute predictions
    ids, pred_probs = classify_lcs(pdf, model, 'cpu')

    # Reformat and re-index
    preds_df = reformat_to_df(pred_probs, ids=ids)
    preds_df.index = preds_df.SNID

    # Take only probabilities to be Ia
    to_return = np.zeros(len(jd), dtype=float)
    ia = preds_df.reindex([str(i) for i in candid[mask].values])
    to_return[mask] = ia.prob_class0.values

    # return probabilities to be Ia
    return pd.Series(to_return)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    model_path = '{}/data/models/snn_models/snn_sn_vs_all/model.pt'.format(path)
    globs["model_path"] = model_path

    # Run the test suite
    spark_unit_tests(globs)
