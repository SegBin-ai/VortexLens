from google.cloud import storage

bucket_name = 'factory-work'


def upload_to_gcs(file_name, file_content, title, description, structure):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Create a folder for the structure and upload the video into that folder
    folder_path = f"{structure}/{file_name}"
    blob = bucket.blob(folder_path)
    
    metadata = {
        'title': title,
        'description': description,
        'structure': structure
    }
    
    blob.metadata = metadata
    content_type = 'video/mp4' if file_name.endswith('.mp4') else 'video/quicktime'
    blob.upload_from_string(file_content, content_type=content_type)
    return blob.public_url



def fetch_videos_metadata():
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    videos = []
    for blob in blobs:
        if blob.metadata and 'title' in blob.metadata and 'description' in blob.metadata:
            video_info = {
                'url': blob.public_url,
                'title': blob.metadata['title'],
                'description': blob.metadata['description'],
                'structure': blob.metadata['structure'],
                'content_type': blob.content_type
            }
            videos.append(video_info)
    return videos

def list_folders_in_bucket():
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    folders = set()
    for blob in blobs:
        folder = blob.name.split('/')[0]
        folders.add(folder)
    return list(folders)

def list_videos_in_folder( folder_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)
    videos = []
    for blob in blobs:
        if blob.name.endswith(('.mp4', '.mov')):
            videos.append({
                'url': blob.public_url,
                'title': blob.metadata.get('title', 'No Title'),
                'description': blob.metadata.get('description', 'No Description'),
                'structure': blob.metadata.get('structure', 'N/A')
            })
    return videos

def get_all_structures():
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    structures = set()
    for blob in blobs:
        structure = blob.name.split('/')[0]
        structures.add(structure)
    return list(structures)