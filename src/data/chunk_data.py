import pandas as pd
import numpy as np

def chunk_data(df, years=5):

    sites = df.index.unique()

    chunk_size = years * 365

    chunks_per_site = {}
    chunks = []
    chunk_idx = 0

    # Iterate over sites
    for site in sites:
        # Calculate number of chunks for the site
        num_chunks = len(df[df.index == site]) // chunk_size
        chunks_per_site[site] = num_chunks
        site_size = len(df[df.index == site])

        # Create a list of chunk indices for each site
        for i in range(num_chunks):
            chunks.append([chunk_idx] * chunk_size)
            chunk_idx += 1

        # Record the leftover data points
        leftover = site_size % chunk_size
        if leftover > 0:
            chunks.append([np.nan] * leftover)

    flattened_chunks = [item for sublist in chunks for item in sublist]

    # Add the chunk indices to the dataframe
    return df.assign(chunk_id=flattened_chunks)