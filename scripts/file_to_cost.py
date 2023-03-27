#
# Generates a CSV giving the cost for the files in the directory in which this script is run.
# The name of the file will be token_costs.csv
# The schema will be filename,cost

import os
import sys
import tiktoken
import argparse

model = "gpt-3.5-turbo"
csv_filename = "token_costs.csv"



# Taken from the OpenAI cookbooks
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    try:
# Like NLTK, the first time this runs causes data to be downloaded from the web.
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        return num_tokens_from_messages(messages, model="cl100k_base")

    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Using gpt-3.5-turbo-0301")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")

    elif model == "gpt-3.5-turbo-0301":
        # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4 
        # If there's a name, the role is omitted.
        tokens_per_name = -1 
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for {model}")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3 # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

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
    if file_exists(csv_filename) and not args.overwrite:
        print(f"ABORTING: File {csv_filename} already exists. Manually delete the file before running the script or pass the --overwrite flag")
        sys.exit(1)
    if args.progress:
        display_progress = progress_generator(len(os.listdir()))
    with open(csv_filename, "w") as output:
        output.write("filename,cost\n")
        for filename in os.listdir():
            # Don't calculate the cost for the csv.
            if filename == csv_filename:
                continue
            if args.progress:
                next(display_progress)
            if not os.path.isfile(filename):
                continue
            with open(filename, "r") as f:
                try:
                    token_cost = num_tokens_from_messages([{
                        "role": "user",
                        "name": "test-user",
                        "content": f.read(),
                        }])
                except UnicodeDecodeError:
                    print(f"\nBad unicode in {filename}")
                    continue
            output.write(f"{filename},{token_cost}\n")
        if args.progress:
            next(display_progress)
