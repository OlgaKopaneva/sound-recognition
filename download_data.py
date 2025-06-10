from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

files_to_download = {
    "FSDKaggle2018.audio_train.zip": "https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_train.zip?download=1",
    "FSDKaggle2018.audio_test.zip": "https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_test.zip?download=1",
    "FSDKaggle2018.meta.zip": "https://zenodo.org/records/2552860/files/FSDKaggle2018.meta.zip?download=1",
}

dir_to_save = Path("data")


def download_file(url, dest):
    response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    total_size = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as file, tqdm(
        desc=dest.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def unzip_file(zip_path, extract_to):
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def download_data():
    for filename, url in files_to_download.items():
        zip_path = dir_to_save / filename
        download_file(url, zip_path)
        unzip_file(zip_path, dir_to_save)
        zip_path.unlink()

    print("ОК, data saved")


if __name__ == "__main__":
    download_data()
