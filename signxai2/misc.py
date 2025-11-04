import requests
from PIL import Image

def get_image(url, path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # raise an error if the download failed

    with open(path, "wb") as f:
        f.write(response.content)

    print("Download complete.")

    return Image.open(path).convert('RGB')

def get_example_image(num):
    if num == 1:
        return get_image("https://upload.wikimedia.org/wikipedia/commons/8/80/Dornbusch_Leuchtturm_1.JPG", "example_1.jpg")
    elif num == 2:
        return get_image("https://upload.wikimedia.org/wikipedia/commons/4/42/North_african_ostrich_%28Struthio_camelus_camelus%29_in_Morocco.jpg", "example_2.jpg")
    else:
        raise Exception('No example defined for num = {}'.format(num))
