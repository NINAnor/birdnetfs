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

    # Limit the total number of segments per species across all files
    num_segments_per_species = config["NUM_SEGMENTS"]
    sampled_df = (
        parquet_df.groupby("species")
        .apply(lambda x: x.sample(min(len(x), num_segments_per_species), random_state=None))
        .reset_index(drop=True)
    )

    # Save the globally sampled DataFrame to a new parquet file
    sampled_df.to_parquet(config["TO_EXTRACT_FILE"])
    print(f"Global sampling complete. Saved to {config['TO_EXTRACT_FILE']}")
