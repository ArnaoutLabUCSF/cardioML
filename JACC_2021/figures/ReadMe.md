from Bio import Entrez, Medline
from collections import Counter, defaultdict
from colour import Color
from glob import glob
from itertools import combinations
from matplotlib import pyplot as plt
from os.path import expanduser, isfile
from places import states, countries, region_country_hash, state_ppn_hash, country_ppn_hash
from string import *
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio


"""
You will need Python 3, the above packages, and two external files: 

1. pubmed_results_parsed.csv (search-result data)
2. arxiv.results.csv (search-result data)
3. places.py (for geotagging)

Instructions: 
1) make sure you're in the directory that contains the file
2) run "python jacc_review_figures.py"
	NOTE: for some, this will be "python3 jacc_review_figures.py"

3) Save all geotagging figures as they appear
4) For the chord diagrams, load chord_country.html and chord_state.html and print as PDFs
