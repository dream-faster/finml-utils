import time

import numpy as np
import pandas as pd
import plotly.express as px
from finml_utils.dataframes import concat_on_index_without_duplicates


def measure_execution_time():
    original = []
    sizes = []
    vectorized_fast = []
    num_dfs = 2
    for num_rows, num_cols in [(10, 3), (1000, 100), (10000, 1000), (1000000, 1000)]:
        dfs = [
            pd.DataFrame(
                np.random.randint(
                    0, 1, size=(num_rows + np.random.randint(0, 5), num_cols)
                )
            )
            for _ in range(num_dfs)
        ]

        dfs[0].iloc[
            np.random.randint(0, num_rows, 10), np.random.randint(0, num_cols, 10)
        ] = np.nan
        # start_time = time.time()
        # concat_on_index_without_duplicates(dfs, keep="last", vectorized="combine_first")
        # print("combine_vectorized: --- %s originals ---" % (time.time() - start_time))
        # combine_first.append(time.time() - start_time)

        # start_time = time.time()
        # concat_on_index_without_duplicates(dfs, keep="last", vectorized="vectorized")
        # print(f"Vectorized:  {time.time() - start_time}")
        # vectorized.append(time.time() - start_time)

        start_time = time.time()
        concat_on_index_without_duplicates(
            dfs, keep="last", vectorized="vectorized_fast"
        )
        print(f"Vectorized Fast:  {time.time() - start_time}")
        vectorized_fast.append(time.time() - start_time)

        start_time = time.time()
        concat_on_index_without_duplicates(dfs, keep="last", vectorized="original")
        print(f"Original: {time.time() - start_time}")
        original.append(time.time() - start_time)
        sizes.append(num_rows * num_cols)

    px.line(
        pd.DataFrame(
            {
                # "vectorized": vectorized,
                "original": original,
                "vectorized_fast": vectorized_fast,
            },
            index=sizes,
        )
    ).show()


measure_execution_time()
