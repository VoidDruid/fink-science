from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType, DoubleType

import pandas as pd
import numpy as np
import light_curve as lc

import os

from fink_science import __file__


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def extract_features_ztf(magpsf, jd, sigmapsf, fid) -> pd.Series:
    amplitude = lc.Amplitude()
    extractor = lc.Extractor(amplitude)

    jd_arr = jd.to_numpy().astype(float)


    result = extractor(jd_arr, magpsf, sigmapsf)
    return pd.Series([1] * 100)


if __name__ == "__main__":
    """ Execute the test suite """

    # TODO: test suite
