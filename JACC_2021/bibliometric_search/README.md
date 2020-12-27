# article_search

## If you use code or data from this repository, please cite Quer G, Arnaout R, Henne M, Arnaout R. Machine Learning and the Future of Cardiovascular Care. Journal of the American College of Cardiology; 26 January 2021.

This code inputs either a Boolean search or, a list of article IDs, and returns in .csv format Title, Author, Abstract, and other information available via Pubmed and arXiv/bioRxiv/medRxiv APIs. Three .csv files are output:

- 'output_searchlog.csv' : a log of the search terms/search IDs used
- 'all_results.csv' : a log of all search results wtih Author, Title, Abstract, regardless of website searched (includes Pubmed and all arXiv sites)
- 'pubmed_results.csv' : a log of pubmed search results only. Additional information is available from the pubmed API.

A 'database.sqlite' file and a log director with log textfiles are also created. 


## Installation

1. Make sure Python 3.6 or higher, pip and git are installed. For example, on Ubuntu run the commands below in a terminal.

```
sudo apt install -y python3
sudo apt install -y python3-pip
sudo apt install -y git
```

2. Open a terminal window
3. Run the commands below. Depending on your system you may need run `pip` instead of `pip3`.

```
git clone https://github.com/ArnaoutLabUCSF/cardioML/bibliometric_search.git
cd bibliometric_search
pip3 install arxiv
pip3 install lxml
pip3 install wget
pip3 install xmltodict
```

## Instructions

1. Open a terminal window. cd to the directory containing `article-search.py`.
2. Optionally, edit the `options.ini` file to your liking, including setting the desired destination folder for the outputs.
3. Edit `input_search_terms.txt` to contain your desired search terms. Each line is a search term.
4. Run `python3 article_search.py -w input_website_files/input_websites.txt -s input_search_terms/input_search_terms.txt -d ~/Desktop/WebSearch`. The `-d` argument allows you to resume a partially completed run of this app.
5. Depending on your system you may need run `python` instead of `python3`.
6. To run multiple instances at the same time, run `bash run.sh`. That will run one instance for each website. That means it will finish four time faster.
7. To re-run a search from scratch, do not use the -d argument and also, delete the sqlite file created in this directory.

## Command line parameters

- `-w`: file containing the list of websites. Default: `input_websites.txt`.
- `-s`: file containing the list of search terms. Default: the value in the `[search terms]` section in `options.ini`.
- `-d`: where to write the logs. Default: `~/Desktop/WebSearch_(current date)`.
- `-i`: if this parameter is present, the script will download the article id's in the id list files specified in `options.ini`. It can be simply `-i`. Nothing needs to follow it. Default is off.

## Options

`options.ini` accepts the following options:

- `maximumResultsPerKeyword`: How many records to download for a given site/keyword combination. -1 means no limit. Default 25000.

### Search terms section

```
[search terms]
(site name in lowercase)=(some file containing the search terms)
```

Example:

```
[search terms]
pubmed=input_search_terms/input_search_terms_pubmed.txt
biorxiv=input_search_terms/input_search_terms_biorxiv_medrxiv.txt
arxiv=input_search_terms/input_search_terms_arxiv.txt
medrxiv=input_search_terms/input_search_terms_biorxiv_medrxiv.txt
```

### ID lists section

```
[id lists]
(site name in lowercase)=(some file containing the article id's)
```

Example:

```
[id lists]
pubmed=input_id_list_files/id_list_pubmed.txt
biorxiv=input_id_list_files/id_list_biorxiv.txt
arxiv=input_id_list_files/id_list_arxiv.txt
medrxiv=input_id_list_files/id_list_medrxiv.txt
```

Code developed by R. Arnaout and Upwork.