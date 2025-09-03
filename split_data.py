import json
import random
from sklearn.model_selection import train_test_split


def filter_and_prepend_path(annotations, filter_prefix, prepend_path):
    """
    Filters annotations based on a path prefix and prepends a new path.

    Args:
        annotations (list): A list of annotation dictionaries.
        filter_prefix (str): The prefix that the 'path' value must have to be included.
        prepend_path (str): The path to prepend to the 'path' value.

    Returns:
        list: The filtered and modified list of annotations.
    """
    filtered_annotations = [
        item for item in annotations if item.get("path", "").startswith(filter_prefix)
    ]

    for item in filtered_annotations:
        item["path"] = prepend_path + item["path"]

    return filtered_annotations


def split_json_data(input_json_path, train_output_path, test_output_path, valid_output_path,
                    filter_prefix="/LibriSpeech", prepend_path="", train_size=0.8, random_state=42):
    """
    Filters, prepends a path to, and splits the annotation data from a JSON file into
    training, testing, and validation sets, and saves them to new JSON files.

    Args:
        input_json_path (str): The path to the input JSON file.
        train_output_path (str): The path to save the training data JSON file.
        test_output_path (str): The path to save the testing data JSON file.
        valid_output_path (str): The path to save the validation data JSON file.
        filter_prefix (str, optional): The required starting string for the 'path' value. Defaults to "/LibriSpeech".
        prepend_path (str, optional): The string to prepend to the 'path' value. Defaults to "".
        train_size (float, optional): The proportion of the dataset for the training set. Defaults to 0.8.
        random_state (int, optional): The seed for the random number generator. Defaults to 42.
    """
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {input_json_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {input_json_path}.")
        return

    annotations = data.get("annotation", [])
    if not annotations:
        print("No 'annotation' key found in the JSON file or it is empty.")
        return

    # Filter and prepend before splitting
    modified_annotations = filter_and_prepend_path(annotations, filter_prefix, prepend_path)

    if not modified_annotations:
        print("No annotations remaining after filtering. No output files will be created.")
        return

    # Split the modified data into training and the rest
    train_data, temp_data = train_test_split(modified_annotations, train_size=train_size, random_state=random_state)

    # Split the rest into testing and validation
    test_data, valid_data = train_test_split(temp_data, test_size=0.5, random_state=random_state)

    # Create the output dictionaries
    train_output = {"annotation": train_data}
    test_output = {"annotation": test_data}
    valid_output = {"annotation": valid_data}

    # Write the output files
    with open(train_output_path, 'w') as f:
        json.dump(train_output, f, indent=4)
    with open(test_output_path, 'w') as f:
        json.dump(test_output, f, indent=4)
    with open(valid_output_path, 'w') as f:
        json.dump(valid_output, f, indent=4)

    print(f"Successfully filtered, modified, and split the data into {len(train_data)} training samples, "
          f"{len(test_data)} testing samples, and {len(valid_data)} validation samples.")


if __name__ == '__main__':
    # Create a dummy json file for demonstration
    dummy_data = {
        "annotation": [
                          {
                              "path": "/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac",
                              "text": "This path starts correctly.",
                              "task": "asr"
                          },
                          {
                              "path": "/OtherData/some_file.flac",
                              "text": "This path should be filtered out.",
                              "task": "asr"
                          }
                      ] * 50  # Replicating the entries to have a list of 100
    }
    with open('annotations.json', 'w') as f:
        json.dump(dummy_data, f, indent=4)

    # --- Configuration ---
    # Define the input and output file paths
    input_json = '/home/lewis/github/SALMONN_support/salmonn_data/salmonn_data/salmonn_stage1_data.json'
    train_json = '/home/lewis/github/SALMONN_support/salmonn_data/salmonn_data/stage1/train.json'
    test_json = '/home/lewis/github/SALMONN_support/salmonn_data/salmonn_data/stage1/test.json'
    valid_json = '/home/lewis/github/SALMONN_support/salmonn_data/salmonn_data/stage1/valid.json'

    # Set your custom path to prepend here
    custom_path_to_prepend = "/home/lewis/github/SALMONN_support/salmonn_data/salmonn_data/stage1/raw"

    # --- End of Configuration ---

    # Split the data with filtering and prepending
    split_json_data(input_json, train_json, test_json, valid_json, prepend_path=custom_path_to_prepend)