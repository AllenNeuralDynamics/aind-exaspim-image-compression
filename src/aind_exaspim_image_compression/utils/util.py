"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from google.cloud import storage
from random import sample

import boto3
import json
import os
import shutil


# --- OS utils ---
def mkdir(path, delete=False):
    """
    Creates a directory at "path".

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. Default is False.

    Returns
    -------
    None
    """
    if delete:
        rmdir(path)
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    """
    Removes directory and all subdirectories at "path".

    Parameters
    ----------
    path : str
        Path to directory and subdirectories to be deleted if they exist.

    Returns
    -------
    None
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def list_dir(directory, extension=None):
    """
    Lists filenames in the given directory. If "extension" is provided,
    filenames ending with the given extension are returned.

    Parameters
    ----------
    directory : str
        Path to directory to be searched.
    extension : str, optional
       Extension of filenames to be returned. Default is None.

    Returns
    -------
    List[str]
        Filenames in the given directory.
    """
    if extension is None:
        return [f for f in os.listdir(directory)]
    else:
        return [f for f in os.listdir(directory) if f.endswith(extension)]


def list_subdirectory_names(directory_path):
    """
    Lists the names of all subdirectories in the given directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory to search.

    Returns
    -------
    subdir_names : List[str]
        List of the names of subdirectories.
    """
    subdir_names = list()
    for d in os.listdir(directory_path):
        path = os.path.join(directory_path, d)
        if os.path.isdir(path) and not d.startswith("."):
            subdir_names.append(d)
    return subdir_names


def list_paths(directory, extension=""):
    """
    Lists all paths within "directory" that end with "extension" if provided.

    Parameters
    ----------
    directory : str
        Directory to be searched.
    extension : str, optional
        If provided, only paths of files with the extension are returned.
        Default is an empty string.

    Returns
    -------
    List[str]
        List of all paths within "directory".

    """
    paths = list()
    for f in os.listdir(directory):
        if f.endswith(extension):
            paths.append(os.path.join(directory, f))
    return paths


# --- IO utils ---
def read_txt(path):
    """
    Reads txt file stored at "path".

    Parameters
    ----------
    path : str
        Path where txt file is stored.

    Returns
    -------
    str
        Contents of txt file.
    """
    with open(path, "r") as f:
        return f.read().splitlines()


def read_json(path):
    """
    Reads JSON file stored at "path".

    Parameters
    ----------
    path : str
        Path where json file is stored.

    Returns
    -------
    str
        Contents of JSON file.
    """
    with open(path, "r") as file:
        return json.load(file)


def write_json(path, my_dict):
    """
    Writes the contents in the given dictionary to a json file at "path".

    Parameters
    ----------
    path : str
        Path where JSON file is stored.
    my_dict : dict
        Dictionary to be written to a JSON.

    Returns
    -------
    None
    """
    with open(path, "w") as file:
        json.dump(my_dict, file, indent=4)


def write_list(path, my_list):
    """
    Writes each item in a list to a text file, with each item on a new line.

    Parameters
    ----------
    path : str
        Path where text file is to be written.
    my_list
        The list of items to write to the file.

    Returns
    -------
    None
    """
    with open(path, "w") as file:
        for item in my_list:
            file.write(f"{item}\n")


# --- GCS utils ---
def copy_gcs_file(bucket_name, source_path, destination_path):
    """
    Copies a file within a GCS bucket.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    source_path : str
        Path to the source file within the bucket.
    destination_path : str
        Path where the source file will be copied to within the same bucket.

    Returns
    -------
    None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    source_blob = bucket.blob(source_path)
    bucket.copy_blob(source_blob, bucket, destination_path)


def copy_gcs_directory(bucket_name, source_prefix, destination_prefix):
    """
    Copies all files from one directory prefix to another within the same GCS
    bucket.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    source_prefix : str
        Prefix of the source directory to copy from.
    destination_prefix : str
        Prefix where files will be copied to.

    Returns
    -------
    None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=source_prefix)
    for blob in blobs:
        new_blob_name = blob.name.replace(source_prefix, destination_prefix, 1)
        bucket.copy_blob(blob, bucket, new_blob_name)


def find_subprefix_with_keyword(bucket_name, prefix, keyword):
    """
    Finds the first GCS subprefix under a given prefix that contains a
    specified keyword.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    prefix : str
        The prefix to search under.
    keyword : str
        Keyword to look for within the subprefixes.

    Returns
    -------
    str
        First subprefix containing the keyword.
    """
    for subprefix in list_gcs_subprefixes(bucket_name, prefix):
        if keyword in subprefix:
            return subprefix
    raise Exception(f"Prefix with keyword '{keyword}' not found in {prefix}")


def get_gcs_directory_size(bucket_name, prefix):
    """
    Calculate the total size of a GCS "directory" (i.e., objects under a prefix),
    and return it in gigabytes (GB).

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    prefix : str
        GCS path prefix (e.g., 'my_folder/' to list everything under that directory).

    Returns
    -------
    float
        Total size in gigabytes.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return sum(blob.size for blob in blobs) / (1024 ** 3)


def list_block_paths(brain_id):
    """
    Lists the GCS paths to image blocks associated with a given brain ID.

    Parameters
    ----------
    brain_id : str
        Unique identifier for a brain dataset.

    Returns
    -------
    img_paths : List[str]
        List of GCS paths (gs://...) pointing to the image blocks.
    """
    # Find prefix containing blocks
    bucket_name = "allen-nd-goog"
    prefix = find_subprefix_with_keyword(bucket_name, "from_aind/", brain_id)
    prefix += "blocks/"

    # Iterate over blocks
    img_paths = list()
    for block_prefix in list_gcs_subprefixes("allen-nd-goog", prefix):
        img_path = find_subprefix_with_keyword(
            bucket_name, block_prefix, "input"
        )
        img_paths.append(f"gs://{bucket_name}/{img_path}")
    return img_paths


def list_gcs_filenames(gcs_dict, extension):
    """
    Lists filenames in a GCS bucket path filtered by file extension.

    Parameters
    ----------
    gcs_dict : dict
        Dictionary with keys "bucket_name" and "path" specifying the GCS
        location.
    extension : str
        File extension to filter by (e.g., '.tif').

    Returns
    -------
    List[str]
        List of blob names containing the specified extension.
    """
    bucket = storage.Client().bucket(gcs_dict["bucket_name"])
    blobs = bucket.list_blobs(prefix=gcs_dict["path"])
    return [blob.name for blob in blobs if extension in blob.name]


def list_gcs_subprefixes(bucket_name, prefix):
    """
    Lists all direct subdirectories of a given prefix in a GCS bucket.

    Parameters
    ----------
    bucket : str
        Name of GCS bucket to be read from.
    prefix : str
        Path to directory in the GCS bucket.

    Returns
    -------
    subdirs : List[str]
         List of direct subdirectories.
    """
    # Load blobs
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter="/"
    )
    [blob.name for blob in blobs]

    # Parse directory contents
    prefix_depth = len(prefix.split("/"))
    subdirs = list()
    for prefix in blobs.prefixes:
        is_dir = prefix.endswith("/")
        is_direct_subdir = len(prefix.split("/")) - 1 == prefix_depth
        if is_dir and is_direct_subdir:
            subdirs.append(prefix)
    return subdirs


def upload_directory_to_gcs(bucket_name, source_dir, destination_dir):
    """
    Uploads all files from a local directory to a GCS bucket, preserving the
    directory structure relative to the source directory.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    source_dir : str
        Path to the local source directory to upload.
    destination_dir : str
        Destination prefix in the GCS bucket where files will be uploaded.

    Returns
    -------
    None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(source_dir):
        for filename in files:
            local_path = os.path.join(root, filename)

            # Compute the relative path and GCS destination path
            path = os.path.relpath(local_path, start=source_dir)
            blob_path = os.path.join(destination_dir, path).replace("\\", "/")

            # Upload the file
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)


# --- S3 utils ---
def exists_in_prefix(bucket_name, prefix, name):
    """
    Checks if a given filename is in a prefix.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to search.
    prefix : str
        S3 prefix to search within.
    name : str
        Filename to search for.

    Returns
    -------
    bool
        Indiciation of whether a given file is in a prefix.
    """
    prefixes = list_s3_prefixes(bucket_name, prefix)
    return sum([1 for prefix in prefixes if name in prefix]) > 0


def list_s3_prefixes(bucket_name, prefix):
    """
    Lists all immediate subdirectories of a given S3 path (prefix).

    Parameters
    -----------
    bucket_name : str
        Name of the S3 bucket to search.
    prefix : str
        S3 prefix to search within.

    Returns
    -------
    List[str]
        List of immediate subdirectories under the specified prefix.
    """
    # Check prefix is valid
    if not prefix.endswith("/"):
        prefix += "/"

    # Call the list_objects_v2 API
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(
        Bucket=bucket_name, Prefix=prefix, Delimiter="/"
    )
    if "CommonPrefixes" in response:
        return [cp["Prefix"] for cp in response["CommonPrefixes"]]
    else:
        return list()


def list_s3_bucket_prefixes(bucket_name, keyword=None):
    """
    Lists all top-level prefixes (directories) in an S3 bucket, optionally
    filtering by a keyword.

    Parameters
    -----------
    bucket_name : str
        Name of the S3 bucket to search.
    keyword : str, optional
        Keyword used to filter the prefixes. Default is None.

    Returns
    --------
    prefixes : List[str]
        A list of top-level prefixes (directories) in the S3 bucket. If a
        keyword is provided, only the matching prefixes are returned.
    """
    # Initializations
    prefixes = list()
    continuation_token = None
    s3 = boto3.client("s3")

    # Main
    keyword = keyword.lower()
    while True:
        # Call the list_objects_v2 API
        list_kwargs = {"Bucket": bucket_name, "Delimiter": "/"}
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)

        # Collect the top-level prefixes
        if "CommonPrefixes" in response:
            for prefix in response["CommonPrefixes"]:
                if keyword and keyword in prefix["Prefix"].lower():
                    prefixes.append(prefix["Prefix"])
                elif keyword is None:
                    prefixes.append(prefix["Prefix"])

        # Check if there are more pages to fetch
        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break
    return prefixes


def is_file_in_prefix(bucket_name, prefix, filename):
    """
    Checks if a specific file exists within a given S3 prefix.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to searched.
    prefix : str
        S3 prefix (path) under which to look for the file.
    filename : str
        Name of the file to search for within the specified prefix.

    Returns
    -------
    bool
        Returns "True" if the file exists within the given prefix,
        otherwise "False".
    """
    for sub_prefix in list_s3_prefixes(bucket_name, prefix):
        if filename in sub_prefix:
            return True
    return False


def write_to_s3(local_path, bucket_name, prefix):
    """
    Writes a single file on local machine to an s3 bucket.

    Parameters
    ----------
    local_path : str
        Path to file to be written to s3.
    bucket_name : str
        Name of s3 bucket.
    prefix : str
        Path within s3 bucket.

    Returns
    -------
    None
    """
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket_name, prefix)


# --- Miscellaneous ---
def sample_once(my_container):
    """
    Samples a single element from the given container.

    Parameters
    ----------
    my_container : container
        Container to be sampled from.

    Returns
    -------
    object
        Element sampled from the given container

    """
    return sample(my_container, 1)[0]


def time_writer(t, unit="seconds"):
    """
    Converts a runtime "t" to a larger unit of time if applicable.

    Parameters
    ----------
    t : float
        Runtime.
    unit : str, optional
        Unit of time that "t" is expressed in. Default is "seconds".

    Returns
    -------
    t : float
        Runtime
    unit : str
        Unit of time.
    """
    assert unit in ["seconds", "minutes", "hours"]
    upd_unit = {"seconds": "minutes", "minutes": "hours"}
    if t < 60 or unit == "hours":
        return t, unit
    else:
        t /= 60
        unit = upd_unit[unit]
        t, unit = time_writer(t, unit=unit)
    return t, unit
