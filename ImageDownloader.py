# from bing_image_downloader import downloader
# query_string = "melanoma picture"
# downloader.download(query_string, limit=500,  output_dir='dataset',
# adult_filter_off=True, force_replace=False, timeout=60)

from google_images_download import google_images_download

def downloadImages(keywords,limit=100,print_urls=False):

    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": keywords,
                 "limit": limit, "print_urls": print_urls}

    paths = response.download(arguments)


# downloadImages("melanoma skin")
downloadImages("Basal cell skin",limit=2)

