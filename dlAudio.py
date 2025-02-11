
import json
import yt_dlp

def dlAudio(URL):
    """Downloads the audio of the video in question
    Arguments:
        URL: URL of the YT video to DL
    Returns:
        A tuple with:
        - video title
        - video ID
        - saved audio file path
    """
    save_string = 'dl/%(id)s.%(ext)s'
    ydl_opts = {}
    vid_json = None
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(URL, download=False)
        vid_json = ydl.sanitize_info(info)
    
    vidTitle = vid_json['title']
    vidID = vid_json['id']

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        # See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'outtmpl': save_string
    }
    error_code = None
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([URL])
        # TODO: implement error handling here later, assume it works for now
        
    return vidTitle, vidID, f"dl/{vidID}.m4a"
