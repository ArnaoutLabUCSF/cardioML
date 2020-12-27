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

"""

# infile with our search results data
print("loading results...")
infile = glob( "pubmed_results_parsed.csv" )[0]
df = pd.read_csv(infile)


# FIGURE 1: TIME TRENDS DATA
# NOTE: this can also be done by using pd.value_counts() for the appropriate study years.
# --------------------------
print("loading papers per year (from biorxiv, arxiv, pubmed) for figure 1...")

# Papers per year: bioRxiv
"""
These were obtained manually via the web as follows:
Go to https://www.biorxiv.org/search
Search for "*" with date limits of the form 01/01/xxxx to 12/31/xxx
"""

year_biorxiv_hash = {
	2013:    77,
	2014:   796,
	2015:  1590,
	2016:  4176,
	2017: 10280, # 10292,
	2018: 19864, # 19996,
	2019: 29840, # 30555,
	2020: 25982, # up through 08/05/20 only, because that's when the infile was run
}
year_biorxiv_hash = defaultdict(int, year_biorxiv_hash)


# Paper per year: arXiv
"""
These were obtained manually via the web as follows:
In bash:
for i in range(2005, 2021):
	print("https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=specific_year&date-year=%i&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first" % i)
Manually enter the resulting URLS and copied back the totals
"""

year_arxiv_hash = {
	2005:  45385, # 45392,
	2006:  48576, # 48581,
	2007:  54150, # 54156,
	2008:  56951, # 56960,
	2009:  62232, # 62245,
	2010:  67487, # 67495,
	2011:  73459, # 73479,
	2012:  81535, # 81565,
	2013:  89812, # 89874,
	2014:  94551, # 94638,
	2015: 102019, # 102172,
	2016: 111404, # 111685,
	2017: 121058, # 121833,
	2018: 138796, # 141518,
	2019: 163514, # 181232,
	2020: 133313, # up through 08/05/20 only, because that's when the infile was run; the URL was https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-year=2020&date-filter_by=date_range&date-from_date=2020-01-01&date-to_date=2020-08-05&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first
}
year_axiv_hash = defaultdict(int, year_arxiv_hash) # in case we access other years


# Papers per year: Pubmed
"""
Uses biopython's Entrez API 
"""
email = "" # enter your email here, so Pubmed can contact you if there's a downloading issue (also good etiquette)
if email.strip() == "rarnaout@gmail.com":
	print("Please set email (ca. line 83 of this program), so Pubmed can contact you if there's a downloading issue (also good etiquette)")
	exit()
year_pubmed_hash = {}
for i in range(1982, 2021): # 1982 is the first year in our results (column Study Year)
	Entrez.email = email
	#search_term = "(%s[1au] OR %s[lastau]) AND 2010:2016[dp]" % (query_name, query_name) # just first or last author
	search_term = "%i[dp]" % i # any author position
	handle = Entrez.egquery(term = search_term)
	record = Entrez.read(handle)
	for row in record["eGQueryResult"]:
	    if row["DbName"]=="pubmed":
	        total = row["Count"]
	        year_pubmed_hash[i] = total
year_pubmed_hash = defaultdict(int, year_pubmed_hash) # in case we access other years
print("\ncopy the followingoutput into plots.xlsx, cols A+B")
print("year\ttotal_pubmed_per_year")
for i, j in sorted(list(year_pubmed_hash.items())):
	print("%i\t%s" % (i, j))

# For 2020 up through 08/05/20, did:
# https://pubmed.ncbi.nlm.nih.gov/?term=%28%222020%2F01%2F01%22%5BDate+-+MeSH%5D+%3A+%222020%2F08%2F05%22%5BDate+-+MeSH%5D%29&sort=date
# Result: 1,265,324; add this in manually


# Papers per year: search results
#
# sanity check that the PMIDs are unique; col "Identifiers"
assert len(list(set(df["Identifiers"]))) == len(df)
#
# get no. papers per year
D = defaultdict(int, Counter(df["Study Year"]))
years_results_hash = defaultdict(int)
print("\ncopy the followingoutput into plots.xlsx, cols (A+)E")
print("year\tsearch_results")
for i in range(min(D), max(D)+1):
	years_results_hash[i] = D[i]
	print("%i\t%i" % (i, D[i]))

# Papers per year: search results, just reviews
#
years_reviews_hash = defaultdict(int)
nonoriginal_rows = df.loc[df["Is NOT Original Research"] == "nonoriginal"] # 342 entries
years = nonoriginal_rows["Study Year"]
D = defaultdict(int, Counter(years))
print("\ncopy the followingoutput into plots.xlsx, cols (A+)K")
print("year\treviews_per_year")
for i in range(min(D), max(D)+1):
	years_reviews_hash[i] = D[i]
	print("%i\t%i" % (i, D[i]))


# These hashes contain the data for figure 1.



# FIGURE 2: CONTENT
# -----------------
print("loading data for figure 2...")
"""
Donut plots. This block generates labeled and unlabeled donut plots. The unlabeled plots can be imported into illustration software for quicker and more precise control over the labels.
"""
plt.ion()
#
# get locations data since year x
first_year = 2016
locations = list(df.loc[df["Study Year"] >= first_year]["all_locations"][df.all_locations != 'emptytag']) # 1591 out of 1785 don't have "emptytag"
columns = ("ML_problem", "ML_method", "Disease", "Data_modality")
line_color = "black"
line_width = 2
do_labels = False # run twice, once True, once False
for column in columns:
	row_entries = list(df.loc[df["Study Year"] >= first_year][column])
	D = defaultdict(int)
	for row_entry in row_entries:
		if type(row_entry) == float: continue
		items = row_entry.split("_")
		for item in items: 
			if item == "heart failure": item = "heartfailure"
			D[item] += 1
	# print("%s\tno" % column)
	try: del D['other']
	except: pass
	# for i, j in sorted(D.items(), key=lambda x: x[1], reverse=True):
	# 	print("%s\t%i" % (i, j))
	#
	# donut plot -- from https://medium.com/@krishnakummar/donut-chart-with-python-matplotlib-d411033c960b
	labels, values = zip(*sorted(D.items(), key=lambda x: x[1], reverse=True))
	if do_labels: labels = list(labels)
	else: labels = ["" for i in range(len(labels))]
	values = list(values)
	n = len(labels)
	cmap = matplotlib.cm.get_cmap('Spectral')
	colors = [Color(rgb=cmap(i)[:3]).hex for i in np.linspace(0, 1, n)]
	# print(colors)
	plt.figure()
	explode = [0]*n  # explode a slice if required
	plt.pie( values, explode=explode, labels=labels, colors=colors, shadow=False, wedgeprops = {'linewidth': line_width, 'edgecolor': line_color} ) # autopct='%1.1f%%', <-- for percentages
    #
	#draw a circle at the center of pie to make it look like a donut
	centre_circle = plt.Circle( (0,0), 0.5, color=line_color, fc='white', linewidth=line_width)
	fig = plt.gcf()
	fig.gca().add_artist(centre_circle)
	plt.axis('equal')
	if do_labels==True: plt.savefig(column + "_labeled.pdf", format="pdf")
	else: plt.savefig(column + "_unlabeled.pdf", format="pdf")

# heatmap of disease vs. data modality
print("creating heatmap of disease vs. data for figure 2...")
# 
columns = ("Disease", "Data_modality")
D_diseases = defaultdict(int)
D_modalities = defaultdict(int)
Ds = (D_diseases, D_modalities)
for column, D in zip(columns, Ds):
	row_entries = list(df.loc[df["Study Year"] >= first_year][column])
	for row_entry in row_entries:
		if type(row_entry) == float: continue
		items = row_entry.split("_")
		for item in items: 
			if item == "heart failure": item = "heartfailure"
			D[item] += 1
	# print("%s\tno" % column)
	try: del D['other']
	except: pass
	# for i, j in sorted(D.items(), key=lambda x: x[1], reverse=True):
	# 	print("%s\t%i" % (i, j))

# make matrix: modalities columns, diseases rows
M = np.zeros([len(D_diseases), len(D_modalities)])
diseases = sorted(D_diseases.keys()) # fixes the order
modalities = sorted(D_modalities.keys()) # fixes the order
disease_column = list(df.loc[df["Study Year"] >= first_year]["Disease"])
modality_column = list(df.loc[df["Study Year"] >= first_year]["Data_modality"])
for d, m in zip(disease_column, modality_column):
	if type(d) == float or type(m) == float: continue
	d_items = d.split("_")
	m_items = m.split("_")
	for i in d_items:
		if i == "other": continue
		if i == "heart failure": i = "heartfailure"
		for j in m_items:
			if j == "other": continue
			M[diseases.index(i), modalities.index(j)] += 1

# seaborn clustermap
M_as_df = pd.DataFrame(M)
M_as_df.columns = modalities # add column headers
M_as_df.rename(index={i:disease for i, disease in enumerate(diseases)}, inplace=True) # add row headers
# sns.clustermap(M_as_df, cmap="Spectral_r", cbar_kws={"ticks":[0, 25, 50, 75, 100]})
sns.clustermap(M_as_df, cmap="Spectral_r", cbar_kws={"ticks":[]})
plt.savefig("disease_modality.pdf", format="pdf")



# FIGURE 3: GEOTAGGING
# --------------------
"""
Strategy is to build a network (Graph object) for countries and a separate one for states, and then use these for chord diagrams and choropleths. We get totals for states and countries at the same time as we construct networks.

Labels are printed to the screen.
"""
print("loading geotagging data for figure 3...")

# slight edits to countries
countries += [" USA", "U.S.A.", "U.S.", " US", " UK", "U.K."] # note spaces
try: countries.remove("Georgia") # no Georgia the country in our dataset, only state
except ValueError: pass

# make dictionary of countries by region
abbrevs = states[1::2]
states = states[0::2]
abbrev_state_hash = {abbrev:state for abbrev, state in zip(abbrevs, states)}
state_abbrev_hash = {state:abbrev for abbrev, state in zip(abbrevs, states)}
country_region_hash = {country.lower():region.lower() for region, country_list in region_country_hash.items() for country in country_list}
synonyms_hash = {
	" USA": "United States", # note space
	"U.S.A.": "United States",
	"U.S.": "United States",
	" US": "United States", # note space
	" UK": "United Kingdom", # note space
	"U.K.": "United Kingdom",
}
#
# construct networks and get state, country totals
print("creating state and country data structures for figure 3...")
state_tot =  defaultdict(int) # total for each state
country_tot = defaultdict(int) # total for each country
region_tot = defaultdict(int) # total for each region
#
C = nx.Graph() # country graph
S = nx.Graph() # state graph
C_edgewidths = []
S_edgewidths = []
for location in locations:
	done = []
	loc_state_set = set() # loc = "local"
	loc_country_set = set()
	loc_region_set = set()
	for state in states: # some entries don't have country, so we have to depend on state
		if state in location:
			if state == "Virginia" and "West Virginia" in location: continue # avoids confusing Virginia and West Virginia
			loc_state_set.add(state)
			loc_country_set.add("United States")
	for abbrev in abbrevs:
		if abbrev in location:
			loc_state_set.add(abbrev_state_hash[abbrev])
			loc_country_set.add("United States")
	for country in countries:
		if country.lower() in location.lower():
			try: country = synonyms_hash[country]
			except: pass
			loc_country_set.add(country)
			region = country_region_hash[country.lower()]
			loc_region_set.add(region)
	for i in loc_state_set: state_tot[i] += 1
	for i in loc_country_set: country_tot[i] += 1
	for i in loc_region_set: region_tot[i] += 1
	# update state graph
	if len(loc_state_set) >= 2:
		for i, j in combinations(sorted(list(loc_state_set)), 2):
			if i < j: e = (i, j)
			else: e = (j, i)
			e1, e2 = e
			if S.has_edge(e1, e2): S[e1][e2]['weight'] += 1
			else: S.add_edge(i, j, weight=1)
	# update country graph
	if len(loc_country_set) >= 2:
		for i, j in combinations(sorted(list(loc_country_set)), 2):
			if i < j: e = (i, j)
			else: e = (j, i)
			e1, e2 = e
			if C.has_edge(e1, e2): C[e1][e2]['weight'] += 1
			else: C.add_edge(i, j, weight=1)

# per-capita output
from places import state_ppn_hash, country_ppn_hash, _50_states
tuples = (["state", state_tot, state_ppn_hash], ["country", country_tot, country_ppn_hash])
per_capitas = []
for s, tot, H in tuples:
	per_capita = {}
	for i, j in sorted(tot.items()):
		per_capita[i] = tot[i]/H[i]
	per_capitas.append(per_capita)
state_per_capita, country_per_capita = per_capitas


# create chord diagrams

# state
chord_state_html = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>
body {
  font: 10px sans-serif;
}

svg text{
  text-anchor:middle;
  font-size:18px;
}
svg .values text{
  pointer-events:none;
  stroke-width: 0.5px;
}
.groups:hover{
  cursor:pointer;
  font-weight:bold;
}

</style>
<body>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="http://vizjs.org/viz.v1.1.2.min.js"></script>
<script>

var width=960, height=960
	,outerRadius = Math.min(width, height) * 0.5 - 40
    ,innerRadius = outerRadius - 60; // 30


var names = [
  ['California', 'Massachusetts', 'New York', 'Texas', 'Maryland', 'Illinois', 'Washington', 'Pennsylvania', 'Minnesota', 'Michigan', 'Virginia', 'Ohio', 'Connecticut', 'North Carolina', 'Georgia', 'Florida', 'Indiana', 'South Carolina', 'Missouri', 'Colorado', 'Tennessee', 'Utah', 'West Virginia', 'Oregon', 'Rhode Island', 'Iowa', 'New Jersey', 'Louisiana', 'Wisconsin', 'Arizona', 'Mississippi', 'Kansas', 'Virgin Islands', 'District of Columbia', 'New Hampshire', 'Puerto Rico', 'Alabama', 'Nebraska', 'South Dakota', 'Kentucky', 'Idaho', 'Maine', 'Oklahoma', 'Delaware', 'Arkansas', 'American Samoa', 'Vermont', 'Montana', 'New Mexico']
];


var m = [
"""
nodes = list(S.nodes)
nodes = sorted(nodes, key=lambda x:state_tot[x], reverse=True) # fix the order (as descending)
n = len(nodes)
M = np.zeros((n, n))
for i in range(n): M[i][i] = state_tot[nodes[i]]
for u, v in combinations(nodes, 2):
	w = 0 # init
	try: w = S[u][v]['weight']
	except: pass
	try: w = S[v][u]['weight']
	except: pass
	i = nodes.index(u)
	j = nodes.index(v)
	M[i][j] = w
# print('copy the following and paste it into chord_state.html after "var m ="')
for i in M:
	chord_state_html += "  %s," % list(i)
chord_state_html += """]

//turn matrix into table
var data=[];
m.forEach(function(r,i){ r.forEach(function(c,j){ data.push([i,j,c])})});

var colors = """
# for i in M: print("  %s," % list(i)) # <-- copy the output to chord_state.html (var m =)

from colour import Color
import matplotlib
top_n = 10 # color the top 10 states
cmap = matplotlib.cm.get_cmap('Spectral')
colors = [Color(rgb=cmap(i)[:3]).hex for i in np.linspace(0, 1, top_n)]
# print('\ncopy the following and paste it into chord_state.html after "colors ="')
# print(colors) # <-- copy the output to chord_state.html (colors =)
chord_state_html += str(colors)
chord_state_html += """

var ch = viz.ch().data(data).padding(.05)
	  	  .innerRadius(innerRadius)
	  	  .outerRadius(outerRadius)
	  	  .label(function(d){ return ""})
	  	  .startAngle(1.5*Math.PI)
	  	  .fill(function(d){ return colors[d];});

var svg = d3.select("body").append("svg").attr("height",height).attr("width",width);

svg.append("g").attr("transform", "translate("+width/2+","+height/2+")").call(ch);

// adjust height of frame in bl.ocks.org
d3.select(self.frameElement).style("height", height+"px").style("width", width+"px");
      
</script>
"""
# f = open(expanduser("~") + "/Desktop/chord_state.html", "w")
f = open("chord_state.html", "w")
f.write(chord_state_html)
f.close()


# country
chord_country_html = """<!DOCTYPE html>
<meta charset="utf-8">
<style>
body {
  font: 10px sans-serif;
}

svg text{
  text-anchor:middle;
  font-size:18px;
}
svg .values text{
  pointer-events:none;
  stroke-width: 0.5px;
}
.groups:hover{
  cursor:pointer;
  font-weight:bold;
}

</style>
<body>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="http://vizjs.org/viz.v1.1.2.min.js"></script>
<script>

var width=960, height=960
	,outerRadius = Math.min(width, height) * 0.5 - 40
    ,innerRadius = outerRadius - 60; // 30


var m = [

"""
nodes = list(C.nodes)
nodes = sorted(nodes, key=lambda x:country_tot[x], reverse=True) # fix the order (as descending)
n = len(nodes)
M = np.zeros((n, n))
for i in range(n): M[i][i] = country_tot[nodes[i]]
for u, v in combinations(nodes, 2):
	# print(u, v)
	w = 0 # init
	try: w = C[u][v]['weight']
	except: pass
	try: w = C[v][u]['weight']
	except: pass
	i = nodes.index(u)
	j = nodes.index(v)
	M[i][j] = w
# print('\ncopy the following and paste it into chord_country.html after "var m ="')
# for i in M: print("  %s," % list(i)) # <-- copy the output to chord_country.html (var m =)
for i in M:
	chord_country_html += "  %s," % list(i)
chord_country_html += """
]

//turn matrix into table
var data=[];
m.forEach(function(r,i){ r.forEach(function(c,j){ data.push([i,j,c])})});

var colors = """
from colour import Color
import matplotlib
top_n = 10 # color the top 10 states
cmap = matplotlib.cm.get_cmap('Spectral')
colors = [Color(rgb=cmap(i)[:3]).hex for i in np.linspace(0, 1, top_n)]
# print('copy the following and paste it into chord_country.html after "colors ="')
# print(colors) # <-- copy the output to chord_country.html (colors =)
chord_country_html += str(colors)
chord_country_html += """
var ch = viz.ch().data(data).padding(.05)
	  	  .innerRadius(innerRadius)
	  	  .outerRadius(outerRadius)
	  	  .label(function(d){ return ""})
	  	  .startAngle(1.5*Math.PI)
	  	  .fill(function(d){ return colors[d];});

var svg = d3.select("body").append("svg").attr("height",height).attr("width",width);

svg.append("g").attr("transform", "translate("+width/2+","+height/2+")").call(ch);

// adjust height of frame in bl.ocks.org
d3.select(self.frameElement).style("height", height+"px").style("width", width+"px");
      
</script>
"""
# f = open(expanduser("~") + "/Desktop/chord_country.html", "w")
f = open("chord_country.html", "w")
f.write(chord_state_html)
f.close()


# Choropleth maps

# STATES
for ii, H in enumerate((state_tot, state_per_capita)):
	_states, tots = zip( *sorted(H.items(), key=lambda x: x[1], reverse=True) )
	_states = list(_states)
	tots = list(tots)
	if ii == 1: # for per-capita, eliminate non-50 states
		new_states = []
		new_tots = []
		for state, tot in zip(_states, tots):
			if state in _50_states:
				new_states.append(state)
				new_tots.append(tot)
		_states = new_states
		tots = new_tots
	for state in state_abbrev_hash.keys():
		if state not in _states:
			_states.append(state)
			tots.append(0.)
	_abbrevs = [state_abbrev_hash[i][-2:] for i in _states]
	fig = go.Figure(data=go.Choropleth(
		locations = _abbrevs,
		z = tots,
		locationmode = 'USA-states', # set of locations match entries in `locations`
		colorscale = [[0, '#ffffff'], [1, '#ff0000']],
		marker_line_color = 'black',
		marker_line_width = 3.,
		colorbar_title = "Publications",
	))
	fig.update_layout(
	    title_text = 'Publications by state',
	    geo_scope='usa', # limite map scope to USA
	)
	fig.show(renderer="browser")

# COUNTRIES
"""
We need country codes. Can get them here: https://www.geonames.org/countries/
"""
from places import code_country_hash
country_code_hash = {j:i for i, j in code_country_hash.items()}
for ii, H in enumerate((country_tot, country_per_capita)):
	_countries, tots = zip( *sorted(H.items(), key=lambda x: x[1], reverse=True) )
	_countries = list(_countries)
	tots = list(tots)
	# manual additions
	country_code_hash["Korea"] = country_code_hash["South Korea"]
	country_code_hash["Czech Republic"] = country_code_hash["Czechia"]
	country_code_hash["D.R. Congo"] = country_code_hash["DR Congo"]
	country_code_hash["Congo"] = country_code_hash["DR Congo"]
	country_code_hash["St Lucia"] = country_code_hash["Saint Lucia"]
	country_code_hash["Trinidad"] = country_code_hash["Trinidad and Tobago"]
	#
	_abbrevs = [country_code_hash[i] for i in _countries]
	for _abbrev in code_country_hash.keys():
		if _abbrev not in _abbrevs:
			_abbrevs.append(_abbrev)
			tots.append(1e-9)
	if ii == 1:
		tots = np.log10(tots)		
	fig = go.Figure(data=go.Choropleth(
		locations = _abbrevs,
		z = tots,
		colorscale = [[0, '#ffffff'], [1, '#ff0000']],
		marker_line_color = 'black',
		marker_line_width = 3.,
		colorbar_title = "Publications",
	))
	fig.update_geos(
		projection_type = "natural earth", # https://plot.ly/python/map-configuration/#map-projections
		visible=False,
	)
	fig.update_layout(
	    title_text = 'Publications by country',
	)
	fig.show(renderer="browser")