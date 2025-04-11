import hashlib
import json
import os

from magika import Magika


# Class to hold file metadata
class FileInfo:
    def __init__(self, hash_str, extension, size_mb, mime_type, magika_group):
        self.hash_str: str = hash_str
        self.extension: str = extension
        self.size_mb: float = size_mb
        self.mime_type: str = mime_type
        self.magika_group: str = magika_group

    def to_dict(self):
        return {
            "hash_str": self.hash_str,
            "extension": self.extension,
            "size_mb": self.size_mb,
            "mime_type": self.mime_type,
            "magika_group": self.magika_group,
        }


ignore_files = [".DS_Store", "_.syftperm"]
ignore_extensions = [".pyc"]


def read_file_as_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()


# Function to calculate SHA-256 hash of a file
def calculate_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def ends_with(filename):
    for ext in ignore_extensions:
        if filename.endswith(ext):
            return True
    return False


# Function to list all files recursively and gather metadata
def list_files_with_metadata(directory_path, cache_file="./file_metadata_cache.json"):
    # Load cached metadata if it exists
    cached_data = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as cache:
            cached_data = json.load(cache)
    
    print(f"[HELPER] Loaded {len(cached_data)} cached files")
    print(f"[HELPER] Scanning directory: {directory_path}")
    
    magika = Magika()
    result = {}
    updated_files = 0
    new_files = 0
    
    # First, check all files in the directory
    for root, _, files in os.walk(directory_path):
        print(f"[HELPER] Scanning directory: {root}")
        for file in files:
            print(f"[HELPER] Found file: {file}")
            if file in ignore_files:
                print(f"[HELPER] Ignoring file (in ignore list): {file}")
                continue
            if ends_with(file):
                print(f"[HELPER] Ignoring file (has ignored extension): {file}")
                continue
                
            file_path = os.path.join(root, file)
            print(f"[HELPER] Processing file: {file_path}")
            file_hash = calculate_hash(file_path)
            
            # Check if the file is in the cache and unchanged
            if file_path in cached_data and cached_data[file_path]["hash_str"] == file_hash:
                print(f"[HELPER] Using cached data for: {file_path}")
                result[file_path] = cached_data[file_path]
            else:
                print(f"[HELPER] Calculating new metadata for: {file_path}")
                # Calculate new metadata
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                extension = os.path.splitext(file)[-1].lstrip(".")
                print(f"[HELPER] File extension: {extension}")
                file_bytes = read_file_as_bytes(file_path)
                mime_info = magika.identify_bytes(file_bytes)
                print(f"[HELPER] MIME type: {mime_info.output.mime_type}")
                
                file_info = FileInfo(
                    hash_str=file_hash,
                    extension=extension,
                    size_mb=size_mb,
                    mime_type=mime_info.output.mime_type,
                    magika_group=mime_info.output.group,
                )
                result[file_path] = file_info.to_dict()
                
                if file_path in cached_data:
                    updated_files += 1
                    print(f"[HELPER] Updated file: {file_path}")
                else:
                    new_files += 1
                    print(f"[HELPER] New file: {file_path}")
    
    # Remove entries for files that no longer exist
    removed_files = 0
    for cached_path in list(cached_data.keys()):
        if not os.path.exists(cached_path):
            removed_files += 1
            print(f"[HELPER] Removing non-existent file from cache: {cached_path}")
            if cached_path in result:
                del result[cached_path]
    
    print(f"[HELPER] Found {new_files} new files, updated {updated_files} files, removed {removed_files} files")
    print(f"[HELPER] Total files in result: {len(result)}")
    for path in result:
        print(f"[HELPER] Result file: {path}")
    
    # Save updated metadata to cache
    with open(cache_file, "w") as cache:
        json.dump(result, cache, indent=4)
    
    print(f"[HELPER] Saved {len(result)} files to cache")
    return result
