from pathlib import Path

UPLOAD_DIRECTORY = Path("data/uploaded")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files):
    saved_files = []

    for uploaded_file in uploaded_files:
        destination = UPLOAD_DIRECTORY / uploaded_file.name

        with open(destination, "wb") as file:
            file.write(uploaded_file.getbuffer())

        saved_files.append(destination)

    return saved_files