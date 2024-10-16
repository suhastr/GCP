import os
from google.cloud import vision
from google.cloud import storage  # For interaction with GCS
import re
from PIL import Image
import io
import matplotlib.pyplot as plt

# No need for setting GOOGLE_APPLICATION_CREDENTIALS when using GCP Notebooks

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision_api.json' # When using locally 

WORD = re.compile(r"\w+")

def detect_text_gcs(gcs_image_uri):
    """Detects text in the image stored in Google Cloud Storage."""
    client = vision.ImageAnnotatorClient()

    # Pass the GCS URI to the Vision API
    image = vision.Image()
    image.source.image_uri = gcs_image_uri

    # Perform text detection
    response = client.document_text_detection(image=image)
    texts = response.text_annotations
    ocr_text = []

    # Extract detected text
    for text in texts:
        ocr_text.append(f"{text.description}")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return ocr_text


# Example usage: Image from GCS
bucket_name = "your-bucket-name"
file_name = None # Replace file name example : "read_id.png"
gcs_image_uri = f"gs://{bucket_name}/{file_name}"

# Call the function
detected_text = detect_text_gcs(gcs_image_uri)
print(detected_text[0])  # Print the first block of detected text

# Loop through and print all lines of text
for line in detected_text:
    print(line)

# Visualization - Download the image and display it inline using matplotlib
def display_image_from_gcs(bucket_name, file_name):
    """Download image from GCS and display inline using matplotlib."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download image into memory
    image_content = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_content))
    return image

# Display the image inline using matplotlib
image = display_image_from_gcs(bucket_name, file_name)

# Using matplotlib to display the image inline in the notebook
plt.imshow(image)
plt.axis('off')  # Turn off axis numbers/labels
plt.show()
