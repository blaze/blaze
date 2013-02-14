import requests, urlparse


def run(hdfs_filepath):
    """Ask local WebHDFS for optimal uri to pull file from HDFS."""

    # TODO: Generalize beyond always looking to localhost:50070
    uri = "http://localhost:50070/webhdfs/v1{}?op=OPEN".format(hdfs_filepath)
    try:
        response = requests.request("GET", uri, allow_redirects=False)
        file_uri = str(response.headers['Location'])
    except:
        file_uri = "ERROR"

    return file_uri

