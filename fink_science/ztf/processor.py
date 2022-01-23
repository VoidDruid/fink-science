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
sturct_def = " double,".join(names) + " double"

@pandas_udf(sturct_def, PandasUDFType.SCALAR)
def extract_features_ztf(arr_magpsf, arr_jd, arr_sigmapsf, _) -> pd.DataFrame:
    result_len = len(arr_magpsf)
    index = np.arange(0, result_len)
    results_df = pd.DataFrame(columns=names, index=index)
    empty_result = [None] * len(names)

    extractor = create_extractor()

    for i, magpsf, jd, sigmapsf in zip(index, arr_magpsf, arr_jd, arr_sigmapsf):
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
                result = empty_result
            # otherwise reraise
            else:
                raise

        results_df.loc[i] = result

    return results_df


if __name__ == "__main__":
    """ Execute the test suite """

    # TODO: test suite
