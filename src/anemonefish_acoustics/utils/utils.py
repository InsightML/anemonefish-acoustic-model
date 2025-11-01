def pretty_path(path: str, num_dirs: int = 3) -> str:
    """
    Take an absolute path and return a pretty path.
    - If it's a file: show the last num_dirs directories and the filename
    - If it's a directory: show the current directory and previous (num_dirs-1) directories, ending with /
    
    Args:
        path: The path to prettify
        num_dirs: Number of directories to include (default: 3)
    """
    import os
    
    # Normalize the path to handle different separators and resolve any . or ..
    normalized_path = os.path.normpath(path)
    
    # Check if it's a file or directory
    if os.path.isfile(normalized_path):
        # For files: show the last num_dirs+1 parts (directories + filename)
        parts = normalized_path.split(os.sep)
        if len(parts) >= num_dirs + 1:
            return '/'.join(parts[-(num_dirs + 1):])
        else:
            return '/'.join(parts) if len(parts) > 1 else os.path.basename(normalized_path)
    else:
        # For directories: show the last num_dirs parts + trailing slash
        parts = normalized_path.split(os.sep)
        if len(parts) >= num_dirs:
            return '/'.join(parts[-num_dirs:]) + '/'
        else:
            return '/'.join(parts) + '/' if len(parts) > 1 else os.path.basename(normalized_path) + '/'