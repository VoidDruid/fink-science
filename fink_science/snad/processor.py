import logging
import os

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType, DoubleType, ArrayType

import pandas as pd
import numpy as np
import light_curve as lc

from fink_science import __file__
from fink_science.tester import spark_unit_tests


logger = logging.getLogger(__name__)


def create_extractor():
    return lc.Extractor(
        lc.Amplitude(),
        lc.BeyondNStd(nstd=1),
        lc.LinearFit(),
        lc.Mean(),
        lc.Median(),
        lc.StandardDeviation(),
        lc.Cusum(),
        lc.ExcessVariance(),
        lc.MeanVariance(),
        lc.Kurtosis(),
        lc.MaximumSlope(),
        lc.Skew(),
        lc.Eta(),
        lc.AndersonDarlingNormal(),
        lc.ReducedChi2(),
        lc.InterPercentileRange(quantile=0.1),
        lc.MedianBufferRangePercentage(quantile=0.1),
        lc.PercentDifferenceMagnitudePercentile(quantile=0.1),
        lc.MedianAbsoluteDeviation(),
        lc.PercentAmplitude(),
        lc.EtaE(),
        lc.LinearTrend(),
        lc.StetsonK(),
        lc.WeightedMean(),
        # 0.4, 0.05 and 0.2, 0.1 are 'default' values
        lc.MagnitudePercentageRatio(
            quantile_numerator=0.4,
            quantile_denominator=0.05,
        ),
        lc.MagnitudePercentageRatio(
            quantile_numerator=0.2,
            quantile_denominator=0.1,
        ),
        #lc.OtsuSplit(), - experimental, not using it yet
        #lc.Bins(),
    )


# 'lc.Extrator' can not be pickled, and thus needs to be created inside the udf,
# but we also need the list of names outside the udf
column_names = list(map(lambda n: 'lc_' + n, create_extractor().names))


@pandas_udf(ArrayType(DoubleType()), PandasUDFType.SCALAR)
def extract_features_snad(arr_magpsf, arr_jd, arr_sigmapsf, _) -> pd.DataFrame:
    # TODO: doctests

    results = []
    extractor = create_extractor()
    for magpsf, jd, sigmapsf in zip(arr_magpsf, arr_jd, arr_sigmapsf):
        magpsf = magpsf.astype("float64")
        jd = jd.astype("float64")
        sigmapsf = jd.astype("float64")

        nans = np.isnan(magpsf) | np.isnan(sigmapsf)
        magpsf = magpsf[~nans]
        sigmapsf = sigmapsf[~nans]
        jd = jd[~nans]

        try:
            result = extractor(jd, magpsf, sigmapsf)
        except ValueError as e:
            # log if unknown error, then skip
            if not (
                e.args[0] in (
                    "feature value is undefined for a flat time series", # one of the series is 'flat' (std==0)
                    "t must be in ascending order",  # incorrect ordering
                ) or (
                    "is smaller than the minimum required length" in e.args[0],  # dataset is too small
                )
            ):
                # otherwise log, then skip
                logger.exception(f"Unknown exception in processor '{__file__}/{extract_features_snad.__name__}'")
            results.append(None)
            continue

        results.append(result)

    return pd.Series(results)


if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample
    
    # Run the test suite
    spark_unit_tests(globs)
