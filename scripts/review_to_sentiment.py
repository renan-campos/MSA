"""
This script reads movie review files and returns a CSV.
The CSV has the filename the polarity of the review, and the cost for gpt to label the review.
"""

import argparse
import time
import csv
import os
import sys
import glob
import openai

MODEL = "gpt-3.5-turbo-0301"
CSV_FILENAME = "sentiment_labels.csv"

def file_exists(filename):
    """Predicate that indicates if the file exists."""
    return os.path.exists(filename)

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", help="Shows processing progress information during runtime", action="store_true")
    parser.add_argument("--overwrite", help="Overwrites the csv.", action="store_true")
    return parser.parse_args()

def progress_generator(total_count):
    last_message_length = 0
    for i in range(total_count):
        print(" " * last_message_length, end="\r")
        message = f"Processing {i+1} of {total_count}"
        print(message, end="")
        last_message_length = len(message)
        yield
    print(" " * last_message_length, end="\r")
    print("Done.")
    yield

if __name__ == "__main__":
    args = process_args()

    already_processed = set()
    if file_exists(CSV_FILENAME):
        with open(CSV_FILENAME, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                already_processed.add(row["filename"])

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    with open(CSV_FILENAME, "w") as output:
        output.write("filename,sentiment,cost\n")

        test_files = glob.glob("*.txt")

        if args.progress:
            display_progress = progress_generator(len(test_files))

        for filename in test_files:
            if args.progress:
                next(display_progress)

            if filename in already_processed:
                continue

            with open(filename, "r") as f:
                review_data = f.read()

            wait_time = 1 # Initial wait time in seconds
            while True:
                try:
                    gpt_response = openai.ChatCompletion.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": "Respond with 'postive' or 'negative'"},
                            {"role": "user", "name": "sentiment-tester", "content": review_data},
                        ],
                    )
                except openai.error.RateLimitError:
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
                break # Get out of the infinite loop
            sentiment = gpt_response["choices"][0]["message"]["content"]
            cost = gpt_response["usage"]["total_tokens"]
            output.write(f"{filename},{sentiment},{cost}\n")
