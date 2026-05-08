with open('main.py', 'r') as f:
    content = f.read()

old_download = """        else:
            logger.info("Downloading from S3 to CPU container for transcription")
            s3_client.download_file(bucket, request.s3_key, str(video_path))"""

new_download = """        else:
            logger.info("Downloading from S3 to CPU container for transcription")
            try:
                s3_client.download_file(bucket, request.s3_key, str(video_path))
            except Exception as e:
                err_str = str(e)
                if "404" in err_str or "Not Found" in err_str or "NoSuchKey" in err_str:
                    parts = request.s3_key.split("/")
                    if parts[0] == "youtube-downloads" and len(parts) >= 3:
                        video_id = parts[2]
                        reconstructed_url = f"https://www.youtube.com/watch?v={video_id}"
                        logger.warning(
                            f"S3 key {request.s3_key} returned 404. "
                            f"Re-ingesting from YouTube: {reconstructed_url}"
                        )
                        _download_youtube(reconstructed_url, video_path)
                        logger.info("Re-uploading re-downloaded video to S3")
                        s3_client.upload_file(str(video_path), bucket, request.s3_key)
                    else:
                        raise RuntimeError(
                            f"S3 object not found ({request.s3_key}) and no YouTube URL to recover from. "
                            "Please re-upload the file."
                        ) from e
                else:
                    raise"""

content = content.replace(old_download, new_download)

with open('main.py', 'w') as f:
    f.write(content)
