# acquire.py

import pandas as pd
import requests
from bs4 import BeautifulSoup
import regex as re
import time
import numpy as np



def create_urls(num=5000):
    num_of_repos=num

    page_numbers = [int((i/100)+1) for i in range(0,num_of_repos,100)]
    print(page_numbers)
    urls = [f'https://github.com/search?p={i}&q=cryptography&type=Repositories&per_page=100' for i in page_numbers]

    print(urls)
    return urls


def get_endpoints(url):

    headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
    
    while True:
        response = requests.get(url, headers=headers)
        if response.ok:
            break
        else:
            print('sleeping')
            time.sleep(20)
            continue
    soup = BeautifulSoup(response.text)
    
    print(response.ok)

    endpoints = []
    subgroups = soup.find_all('div', {"class":"f4 text-normal"})

    for group in subgroups:
        endpoints.append(re.search('href=".*"', str(group))[0][6:-1])

    return endpoints


def make_all_endpoints():
    urls = create_urls()
    for url in urls:
        print(url)
    all_endpoints = []

    for i, page in enumerate(urls):
        all_endpoints.append(get_endpoints(page))
        print(page)

    print(len(all_endpoints))

    return all_endpoints

def acquire_endpoints():
    our_endpoints = pd.Series(make_all_endpoints(), name='endpoints')
    our_endpoints.to_csv('endpoints.csv', index=False)

    return our_endpoints

def flatten_endpoints():
    end_points = pd.read_csv('endpoints.csv')
    all_values = []
    for value in end_points.values:
        for ep in value:
            all_values.append(ep)

    final_values = []
    #print(all_values)
    for value in all_values:
        for val in value.split("'"):
            if len(val) > 3:
                final_values.append(val)
                print(val)

    return pd.Series(final_values, name='endpoints')

"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = [repo[1:] for repo in list(pd.read_csv('final_repos.csv').endpoints.values)]


headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }



def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    df = acquire_endpoints()
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)
    print('howdy')