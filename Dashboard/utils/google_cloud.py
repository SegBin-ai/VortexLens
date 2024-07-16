from google.cloud import storage

bucket_name = 'factory-work'

def upload_to_gcs(file_name, file_content, title, description):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    metadata = {
        'title': title,
        'description': description
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
                'content_type': blob.content_type
            }
            videos.append(video_info)
    return videos