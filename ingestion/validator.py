


def validate_files(uploaded_files, allowed_extensions):

    """Validate the uploaded files based on their extensions.
    Args:
        uploaded_files (list): List of uploaded files.
        allowed_extensions (list): List of allowed file extensions.
    Returns:
        list: List of invalid files (if any).
    """
    invalid_files = []

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension not in allowed_extensions:
            invalid_files.append(uploaded_file.name)
    
    return invalid_files