from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType, DoubleType, ArrayType, StructType, StructField, Row

import pandas as pd
import numpy as np
import light_curve as lc

from fink_science import __file__

from fink_science.ztf.utilities import fix_nans


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
        lc.WeightedMean(),
        lc.Eta(),
        lc.AndersonDarlingNormal(),
        lc.ReducedChi2(),
        lc.InterPercentileRange(quantile=0.1),
        #lc.MagnitudePercentageRatio(),
        lc.MedianBufferRangePercentage(quantile=0.1),
        lc.PercentDifferenceMagnitudePercentile(quantile=0.1),
        lc.MedianAbsoluteDeviation(),
        lc.PercentAmplitude(),
        lc.EtaE(),
        lc.LinearTrend(),
        lc.StetsonK(),
        lc.WeightedMean(),
        #lc.Bins(),
        #lc.OtsuSplit(),
    )


# 'lc.Extrator' can not be pickled, and thus needs to be created inside the udf,
# but we also need the list of names outside the udf
names = create_extractor().names


@pandas_udf(ArrayType(DoubleType()), PandasUDFType.SCALAR)
def extract_features_ztf(arr_magpsf, arr_jd, arr_sigmapsf, _) -> pd.Series:
    results = []
    extractor = create_extractor()
    for magpsf, jd, sigmapsf in zip(arr_magpsf, arr_jd, arr_sigmapsf):
        magpsf = magpsf.astype("float64")
        jd = jd.astype("float64")
        sigmapsf = jd.astype("float64")
        fix_nans(magpsf)
        fix_nans(sigmapsf)

        try:
            result = extractor(jd, magpsf, sigmapsf)
        except ValueError as e:
            # skip if
            if e.args[0] in (
                "feature value is undefined for a flat time series", # one of the series is 'flat' (std==0)
                "t must be in ascending order",  # incorrect ordering
            ) or (
                "is smaller than the minimum required length" in e.args[0],  # dataset is too small
            ):
                results.append(None)
                continue
            # otherwise reraise
            raise

        results.append(result)

    return pd.Series(results)


if __name__ == "__main__":
    """ Execute the test suite """

    # TODO: test suite
