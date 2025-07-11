import urllib.request

model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
output_file = "sam_vit_h_4b8939.pth"

print(f"Downloading SAM model checkpoint (2.4GB)...")
urllib.request.urlretrieve(model_url, output_file)
print("Download complete! File saved as:", output_file)