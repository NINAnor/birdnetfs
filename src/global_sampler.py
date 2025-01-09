import pandas as pd
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_connection.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Read the original Parquet file
    parquet_df = pd.read_parquet(config["PARQUET_DB"])

    # Filter for segments where start < 3600
    filtered_df = parquet_df[parquet_df["start"] < 3600]

    # Limit the total number of segments per species across all files
    num_segments_per_species = config["NUM_SEGMENTS"]
    sampled_df = (
        filtered_df.groupby("species")
        .apply(lambda x: x.sample(min(len(x), num_segments_per_species), random_state=None))
        .reset_index(drop=True)
    )

    # Add a unique row identifier
    sampled_df['rowid'] = range(len(sampled_df))

    # Save the globally sampled DataFrame to a new parquet file
    sampled_df.to_parquet(config["TO_EXTRACT_FILE"])
    print(f"Global sampling complete. Saved to {config['TO_EXTRACT_FILE']}")
