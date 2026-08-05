#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging
import os

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading input artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading dataframe")
    df = pd.read_csv(artifact_local_path)

    logger.info(
        "Dropping price outliers outside [%s, %s]",
        args.min_price, args.max_price
    )
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    output_path = "clean_sample.csv"
    logger.info("Saving cleaned data to %s", output_path)
    df.to_csv(output_path, index=False)

    logger.info("Uploading cleaned artifact %s to W&B", args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)

    # Make sure the artifact is fully uploaded before the run ends
    artifact.wait()

    # Clean up local file
    if os.path.isfile(output_path):
        os.remove(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name (with version/tag) of the input W&B artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact to create in W&B",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A brief description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price to consider (rows with lower price will be dropped)",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to consider (rows with higher price will be dropped)",
        required=True,
    )

    args = parser.parse_args()

    go(args)
