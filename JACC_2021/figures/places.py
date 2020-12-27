states = [
"Alabama",
", AL",
"Alaska",
", AK",
"Arizona",
", AZ",
"Arkansas",
", AR",
"California",
", CA",
"Colorado",
", CO",
"Connecticut",
", CT",
"Delaware",
", DE",
"District of Columbia",
", DC",
"Florida",
", FL",
"Georgia",
", GA",
"Hawaii",
", HI",
"Idaho",
", ID",
"Illinois",
", IL",
"Indiana",
", IN",
"Iowa",
", IA",
"Kansas",
", KS",
"Kentucky",
", KY",
"Louisiana",
", LA",
"Maine",
", ME",
"Maryland",
", MD",
"Massachusetts",
", MA",
"Michigan",
", MI",
"Minnesota",
", MN",
"Mississippi",
", MS",
"Missouri",
", MO",
"Montana",
", MT",
"Nebraska",
", NE",
"Nevada",
", NV",
"New Hampshire",
", NH",
"New Jersey",
", NJ",
"New Mexico",
", NM",
"New York",
", NY",
"North Carolina",
", NC",
"North Dakota",
", ND",
"Ohio",
", OH",
"Oklahoma",
", OK",
"Oregon",
", OR",
"Pennsylvania",
", PA",
"Rhode Island",
", RI",
"South Carolina",
", SC",
"South Dakota",
", SD",
"Tennessee",
", TN",
"Texas",
", TX",
"Utah",
", UT",
"Vermont",
", VT",
"Virginia",
", VA",
"Washington",
", WA",
"West Virginia",
", WV",
"Wisconsin",
", WI",
"Wyoming",
", WY",
"American Samoa",
", AS",
"Guam",
", GU",
"Northern Mariana Islands",
", MP",
"Puerto Rico",
", PR",
"Virgin Islands",
", VI",
]

_50_states = [
"Alabama",
"Alaska",
"Arizona",
"Arkansas",
"California",
"Colorado",
"Connecticut",
"Delaware",
"Florida",
"Georgia",
"Hawaii",
"Idaho",
"Illinois",
"Indiana",
"Iowa",
"Kansas",
"Kentucky",
"Louisiana",
"Maine",
"Maryland",
"Massachusetts",
"Michigan",
"Minnesota",
"Mississippi",
"Missouri",
"Montana",
"Nebraska",
"Nevada",
"New Hampshire",
"New Jersey",
"New Mexico",
"New York",
"North Carolina",
"North Dakota",
"Ohio",
"Oklahoma",
"Oregon",
"Pennsylvania",
"Rhode Island",
"South Carolina",
"South Dakota",
"Tennessee",
"Texas",
"Utah",
"Vermont",
"Virginia",
"Washington",
"West Virginia",
"Wisconsin",
"Wyoming",
]

countries = [ # https://gist.github.com/kalinchernev/486393efcca01623b18d; lightly manually edited
	"Afghanistan",
	"Albania",
	"Algeria",
	"Andorra",
	"Angola",
	"Antigua",
	"Argentina",
	"Armenia",
	"Australia",
	"Austria",
	"Azerbaijan",
	"Bahamas",
	"Bahrain",
	"Bangladesh",
	"Barbados",
	"Belarus",
	"Belgium",
	"Belize",
	"Benin",
	"Bhutan",
	"Bolivia",
	"Bosnia Herzegovina",
	"Botswana",
	"Brazil",
	"Brunei",
	"Bulgaria",
	"Burkina Faso",
	"Burundi",
	"Cambodia",
	"Cameroon",
	"Canada",
	"Cape Verde",
	"Central African Republic",
	"Chad",
	"Chile",
	"China",
	"Colombia",
	"Comoros",
	"Congo",
	"D.R. Congo",
	"Costa Rica",
	"Croatia",
	"Cuba",
	"Cyprus",
	"Czech Republic",
	"Denmark",
	"Djibouti",
	"Dominica",
	"Dominican Republic",
	"East Timor",
	"Ecuador",
	"Egypt",
	"El Salvador",
	"Equatorial Guinea",
	"Eritrea",
	"Estonia",
	"Ethiopia",
	"Fiji",
	"Finland",
	"France",
	"Gabon",
	"Gambia",
	"Georgia",
	"Germany",
	"Ghana",
	"Greece",
	"Grenada",
	"Guatemala",
	"Guinea",
	"Guinea-Bissau",
	"Guyana",
	"Haiti",
	"Honduras",
	"Hungary",
	"Iceland",
	"India",
	"Indonesia",
	"Iran",
	"Iraq",
	"Ireland",
	"Israel",
	"Italy",
	"Ivory Coast",
	"Jamaica",
	"Japan",
	"Jordan",
	"Kazakhstan",
	"Kenya",
	"Kiribati",
	"North Korea",
	"South Korea",
	"Korea",
	"Kosovo",
	"Kuwait",
	"Kyrgyzstan",
	"Laos",
	"Latvia",
	"Lebanon",
	"Lesotho",
	"Liberia",
	"Libya",
	"Liechtenstein",
	"Lithuania",
	"Luxembourg",
	"Macedonia",
	"Madagascar",
	"Malawi",
	"Malaysia",
	"Maldives",
	"Mali",
	"Malta",
	"Marshall Islands",
	"Mauritania",
	"Mauritius",
	"Mexico",
	"Micronesia",
	"Moldova",
	"Monaco",
	"Mongolia",
	"Montenegro",
	"Morocco",
	"Mozambique",
	"Myanmar",
	"Namibia",
	"Nauru",
	"Nepal",
	"Netherlands",
	"New Zealand",
	"Nicaragua",
	"Niger",
	"Nigeria",
	"Norway",
	"Oman",
	"Pakistan",
	"Palau",
	"Panama",
	"Papua New Guinea",
	"Paraguay",
	"Peru",
	"Philippines",
	"Poland",
	"Portugal",
	"Qatar",
	"Romania",
	"Russia",
	"Rwanda",
	"St Kitts",
	"St Lucia",
	"Saint Vincent",
	"Samoa",
	"San Marino",
	"Sao Tome",
	"Saudi Arabia",
	"Senegal",
	"Serbia",
	"Seychelles",
	"Sierra Leone",
	"Singapore",
	"Slovakia",
	"Slovenia",
	"Solomon Islands",
	"Somalia",
	"South Africa",
	"South Sudan",
	"Spain",
	"Sri Lanka",
	"Sudan",
	"Suriname",
	"Swaziland",
	"Sweden",
	"Switzerland",
	"Syria",
	"Taiwan",
	"Tajikistan",
	"Tanzania",
	"Thailand",
	"Togo",
	"Tonga",
	"Trinidad",
	"Tunisia",
	"Turkey",
	"Turkmenistan",
	"Tuvalu",
	"Uganda",
	"Ukraine",
	"United Arab Emirates",
	"United Kingdom",
	"United States",
	"Uruguay",
	"Uzbekistan",
	"Vanuatu",
	"Vatican City",
	"Venezuela",
	"Vietnam",
	"Yemen",
	"Zambia",
	"Zimbabwe"
	]

region_country_hash = {
"East Asia": [
	"Brunei",
	"Cambodia",
	"China",
	"Indonesia",
	"Japan",
	"North Korea",
	"South Korea", "Korea",
	"Laos",
	"Malaysia",
	"Mongolia",
	"Myanmar",
	"Philippines",
	"Singapore",
	"Taiwan",
	"Thailand",
	"Vietnam",
],
"South and Central Asia": [
	"Afghanistan",
	"Azerbaijan",
	"Bangladesh",
	"Bhutan",
	"India",
	"Kazakhstan",
	"Kyrgyzstan",
	"Nepal",
	"Pakistan",
	"Sri Lanka",
	"Tajikistan",
	"Turkmenistan",
	"Uzbekistan",
],
"Central America and Caribbean": [
	"Antigua",
	"Bahamas",
	"Barbados",
	"Belize",
	"Costa Rica",
	"Cuba",
	"Dominica",
	"Dominican Republic",
	"El Salvador",
	"Grenada",
	"Guatemala",
	"Haiti",
	"Honduras",
	"Jamaica",
	"Nicaragua",
	"Panama",
	"St Kitts", "Nevis",
	"St Lucia",
	"Saint Vincent", "Grenadines",
	"Trinidad", "Tobago",
],
"Europe": [
	"Albania",
	"Andorra",
	"Armenia",
	"Austria",
	"Belarus",
	"Belgium",
	"Bosnia Herzegovina", "Bosnia-Herzegovina",
	"Bulgaria",
	"Croatia",
	"Cyprus",
	"Czech Republic",
	"Denmark",
	"Estonia",
	"Finland",
	"France",
	"Georgia",
	"Germany",
	"Greece",
	"Hungary",
	"Iceland",
	"Ireland",
	"Italy",
	"Kosovo",
	"Latvia",
	"Liechtenstein",
	"Lithuania",
	"Luxembourg",
	"Macedonia",
	"Malta",
	"Moldova",
	"Monaco",
	"Montenegro",
	"Netherlands",
	"Norway",
	"Poland",
	"Portugal",
	"Romania",
	"Russia",
	"San Marino",
	"Serbia",
	"Slovakia",
	"Slovenia",
	"Spain",
	"Sweden",
	"Switzerland",
	"Turkey",
	"Ukraine",
	"United Kingdom", "UK", "U.K.",
	"Vatican City",
],
"Middle East": [
	"Algeria",
	"Bahrain",
	"Djibouti",
	"Egypt",
	"Iran",
	"Iraq",
	"Israel",
	"Jordan",
	"Kuwait",
	"Lebanon",
	"Libya",
	"Morocco",
	"Oman",
	"Qatar",
	"Saudi Arabia",
	"Sudan",
	"Syria",
	"Tunisia",
	"United Arab Emirates",
	"Yemen",
],
"North America": [
	"Canada",
	"Mexico",
	"United States",
],
"Oceania": [
	"Australia",
	"East Timor",
	"Fiji",
	"Kiribati",
	"Marshall Islands",
	"Micronesia",
	"Nauru",
	"New Zealand",
	"Palau",
	"Papua New Guinea",
	"Samoa",
	"Solomon Islands",
	"Tonga",
	"Tuvalu",
	"Vanuatu",
],
"South America": [
	"Argentina",
	"Bolivia",
	"Brazil",
	"Chile",
	"Colombia",
	"Ecuador",
	"Guyana",
	"Paraguay",
	"Peru",
	"Suriname",
	"Uruguay",
	"Venezuela",
],
"Sub-Saharan Africa": [
	"Angola",
	"Benin",
	"Botswana",
	"Burkina Faso",
	"Burundi",
	"Cameroon",
	"Cape Verde",
	"Central African Republic",
	"Chad",
	"Comoros",
	"Congo",
	"D.R. Congo",
	"Equatorial Guinea",
	"Eritrea",
	"Ethiopia",
	"Gabon",
	"Gambia",
	"Ghana",
	"Guinea",
	"Guinea-Bissau",
	"Ivory Coast",
	"Kenya",
	"Lesotho",
	"Liberia",
	"Madagascar",
	"Malawi",
	"Maldives",
	"Mali",
	"Mauritania",
	"Mauritius",
	"Mozambique",
	"Namibia",
	"Niger",
	"Nigeria",
	"Rwanda",
	"Sao Tome", "Principe",
	"Senegal",
	"Seychelles",
	"Sierra Leone",
	"Somalia",
	"South Africa",
	"South Sudan",
	"Swaziland",
	"Tanzania",
	"Togo",
	"Uganda",
	"Zambia",
	"Zimbabwe",
],
}

state_ppn_hash = {
"California": 39512223,
"Texas": 28995881,
"Florida": 21477737,
"New York": 19453561,
"Pennsylvania": 12801989,
"Illinois": 12671821,
"Ohio": 11689100,
"Georgia": 10617423,
"North Carolina": 10488084,
"Michigan": 9986857,
"New Jersey": 8882190,
"Virginia": 8535519,
"Washington": 7614893,
"Arizona": 7278717,
"Massachusetts": 6949503,
"Tennessee": 6833174,
"Indiana": 6732219,
"Missouri": 6137428,
"Maryland": 6045680,
"Wisconsin": 5822434,
"Colorado": 5758736,
"Minnesota": 5639632,
"South Carolina": 5148714,
"Alabama": 4903185,
"Louisiana": 4648794,
"Kentucky": 4467673,
"Oregon": 4217737,
"Oklahoma": 3956971,
"Connecticut": 3565287,
"Utah": 3205958,
"Iowa": 3155070,
"Puerto Rico": 3193694,
"Nevada": 3080156,
"Arkansas": 3017825,
"Mississippi": 2976149,
"Kansas": 2913314,
"New Mexico": 2096829,
"Nebraska": 1934408,
"Idaho": 1792065,
"West Virginia": 1787147,
"Hawaii": 1415872,
"New Hampshire": 1359711,
"Maine": 1344212,
"Montana": 1068778,
"Rhode Island": 1059361,
"Delaware": 973764,
"South Dakota": 884659,
"North Dakota": 762062,
"Alaska": 731545,
"District of Columbia": 705749,
"Vermont": 623989,
"Wyoming": 578759,
"American Samoa": 56700,
"Virgin Islands": 104909,
}

country_ppn_hash = {
	"China": 1401570000,
	"India": 1359280000,
	"United States": 330852000,
	"Indonesia": 268074600,
	"Brazil": 211172000,
	"Pakistan": 207867000,
	"Nigeria": 200962417,
	"Bangladesh": 168192000,
	"Russia": 146793744,
	"Mexico": 126577691,
	"Japan": 126200000,
	"Philippines": 109047000,
	"Egypt": 100145900,
	"Ethiopia": 98665000,
	"Vietnam": 95354000,
	"Congo": 86727573,
	"D.R. Congo": 86727573,
	"Germany": 82979100,
	"Iran": 83243800,
	"Turkey": 82003882,
	"France": 66992000,
	"Thailand": 66476205,
	"United Kingdom": 66040229,
	"Italy": 60375749,
	"South Africa": 57725600,
	"Tanzania": 55890747,
	"Myanmar": 54339766,
	"Kenya": 52214791,
	"South Korea": 51811167,
	"Korea": 51811167,
	"Colombia": 49849818,
	"Spain": 46733038,
	"Argentina": 44938712,
	"Algeria": 43378027,
	"Ukraine": 42101650,
	"Sudan": 42291492,
	"Uganda": 40006700,
	"Iraq": 39127900,
	"Poland": 38413000,
	"Canada": 37817500,
	"Morocco": 35300100,
	"Uzbekistan": 34028693,
	"Saudi Arabia": 33413660,
	"Malaysia": 33089200,
	"Peru": 32495510,
	"Venezuela": 32219521,
	"Afghanistan": 31575018,
	"Ghana": 30280811,
	"Angola": 30175553,
	"Nepal": 29609623,
	"Yemen": 29579986,
	"Mozambique": 27909798,
	"Ivory Coast": 25823071,
	"North Korea": 25450000,
	"Australia": 25695400,
	"Madagascar": 25263000,
	"Cameroon": 24348251,
	"Taiwan": 23589192,
	"Niger": 22314743,
	"Sri Lanka": 21670112,
	"Burkina Faso": 20870060,
	"Mali": 19973000,
	"Romania": 19523621,
	"Chile": 19107216,
	"Syria": 18499181,
	"Kazakhstan": 18654760,
	"Guatemala": 17679735,
	"Malawi": 17563749,
	"Zambia": 17381168,
	"Netherlands": 17382200,
	"Ecuador": 17435100,
	"Cambodia": 16289270,
	"Senegal": 16209125,
	"Chad": 15692969,
	"Somalia": 15636171,
	"Zimbabwe": 15159624,
	"South Sudan": 12778250,
	"Rwanda": 12374397,
	"Guinea": 12218357,
	"Benin": 11733059,
	"Haiti": 11577779,
	"Tunisia": 11551448,
	"Bolivia": 11469896,
	"Belgium": 11463692,
	"Cuba": 11221060,
	"Burundi": 10953317,
	"Greece": 10741165,
	"Czech Republic": 10649800,
	"Jordan": 10626200,
	"Dominican Republic": 10358320,
	"Portugal": 10291027,
	"Sweden": 10255102,
	"Azerbaijan": 9981457,
	"Hungary": 9778371,
	"United Arab Emirates": 9682088,
	"Belarus": 9465300,
	"Honduras": 9158345,
	"Israel": 9161580,
	"Tajikistan": 8931000,
	"Austria": 8859992,
	"Papua New Guinea": 8558800,
	"Switzerland": 8542323,
	"Sierra Leone": 7901454,
	"Togo": 7538000,
	"Paraguay": 7152703,
	"Laos": 7123205,
	"Serbia": 7001444,
	"Bulgaria": 7000039,
	"El Salvador": 6704864,
	"Libya": 6569864,
	"Nicaragua": 6393824,
	"Kyrgyzstan": 6389500,
	"Lebanon": 6065922,
	"Turkmenistan": 5942561,
	"Denmark": 5811413,
	"Singapore": 5638700,
	"Republic of the Congo": 5542197,
	"Finland": 5518393,
	"Central African Republic": 5496011,
	"Slovakia": 5450421,
	"Norway": 5334762,
	"Eritrea": 5309659,
	"Costa Rica": 5058007,
	"Palestine": 4976684,
	"New Zealand": 5043990,
	"Ireland": 4857000,
	"Oman": 4686829,
	"Liberia": 4475353,
	"Kuwait": 4420110,
	"Panama": 4218808,
	"Croatia": 4105493,
	"Mauritania": 4077347,
	"Georgia": 3723500,
	"Moldova": 3547539,
	"Uruguay": 3518552,
	"Bosnia and Herzegovina": 3502550,
	"Mongolia": 3305346,
	"Armenia": 2962100,
	"Albania": 2862427,
	"Lithuania": 2790472,
	"Qatar": 2772294,
	"Jamaica": 2726667,
	"Namibia": 2458936,
	"Botswana": 2338851,
	"Gambia": 2228075,
	"Gabon": 2109099,
	"Slovenia": 2080908,
	"North Macedonia": 2075301,
	"Lesotho": 2007201,
	"Latvia": 1915100,
	"Guinea-Bissau": 1604528,
	"Bahrain": 1543300,
	"East Timor": 1387149,
	"Trinidad and Tobago": 1359193,
	"Trinidad": 1359193,
	"Equatorial Guinea": 1358276,
	"Estonia": 1324820,
	"Mauritius": 1265577,
	"Eswatini": 1093238,
	"Djibouti": 1078373,
	"Fiji": 884887,
	"Comoros": 873724,
	"Cyprus": 864200,
	"Guyana": 786508,
	"Bhutan": 741672,
	"Solomon Islands": 680806,
	"Montenegro": 622359,
	"Luxembourg": 613894,
	"Suriname": 573085,
	"Cape Verde": 550483,
	"Malta": 475701,
	"Brunei": 421300,
	"Belize": 398050,
	"Bahamas": 385340,
	"Maldives": 378114,
	"Iceland": 358780,
	"Vanuatu": 304500,
	"Barbados": 287010,
	"Sao Tome": 201784,
	"Principe": 201784,
	"Samoa": 200874,
	"St Lucia": 180454,
	"Kiribati": 120100,
	"Saint Vincent and the Grenadines": 110520,
	"Grenada": 108825,
	"Federated States of Micronesia": 105300,
	"Antigua and Barbuda": 104084,
	"Tonga": 100300,
	"Seychelles": 96762,
	"Andorra": 76177,
	"Dominica": 74679,
	"Saint Kitts and Nevis": 56345,
	"Marshall Islands": 55500,
	"Liechtenstein": 38380,
	"Monaco": 38300,
	"San Marino": 33422,
	"Palau": 17900,
	"Nauru": 11000,
	"Tuvalu": 10200,
	"Vatican City": 800
}

code_country_hash = { # ISO-3166
	"AND": "Andorra",
	"ARE": "United Arab Emirates",
	"AFG": "Afghanistan",
	"ATG": "Antigua and Barbuda",
	"AIA": "Anguilla",
	"ALB": "Albania",
	"ARM": "Armenia",
	"ANT": "Netherlands Antilles",
	"AGO": "Angola",
	"ATA": "Antarctica",
	"ARG": "Argentina",
	"ASM": "American Samoa",
	"AUT": "Austria",
	"AUS": "Australia",
	"ABW": "Aruba",
	"ALA": "Åland",
	"AZE": "Azerbaijan",
	"BIH": "Bosnia and Herzegovina",
	"BRB": "Barbados",
	"BGD": "Bangladesh",
	"BEL": "Belgium",
	"BFA": "Burkina Faso",
	"BGR": "Bulgaria",
	"BHR": "Bahrain",
	"BDI": "Burundi",
	"BEN": "Benin",
	"BLM": "Saint Barthélemy",
	"BMU": "Bermuda",
	"BRN": "Brunei",
	"BOL": "Bolivia",
	"BES": "Bonaire, Sint Eustatius, and Saba",
	"BRA": "Brazil",
	"BHS": "Bahamas",
	"BTN": "Bhutan",
	"BVT": "Bouvet Island",
	"BWA": "Botswana",
	"BLR": "Belarus",
	"BLZ": "Belize",
	"CAN": "Canada",
	"CCK": "Cocos [Keeling] Islands",
	"COD": "DR Congo",
	"CAF": "Central African Republic",
	"COG": "Congo Republic",
	"CHE": "Switzerland",
	"CIV": "Ivory Coast",
	"COK": "Cook Islands",
	"CHL": "Chile",
	"CMR": "Cameroon",
	"CHN": "China",
	"COL": "Colombia",
	"CRI": "Costa Rica",
	"SCG": "Serbia and Montenegro",
	"CUB": "Cuba",
	"CPV": "Cabo Verde",
	"CUW": "Curaçao",
	"CXR": "Christmas Island",
	"CYP": "Cyprus",
	"CZE": "Czechia",
	"DEU": "Germany",
	"DJI": "Djibouti",
	"DNK": "Denmark",
	"DMA": "Dominica",
	"DOM": "Dominican Republic",
	"DZA": "Algeria",
	"ECU": "Ecuador",
	"EST": "Estonia",
	"EGY": "Egypt",
	"ESH": "Western Sahara",
	"ERI": "Eritrea",
	"ESP": "Spain",
	"ETH": "Ethiopia",
	"FIN": "Finland",
	"FJI": "Fiji",
	"FLK": "Falkland Islands",
	"FSM": "Micronesia",
	"FRO": "Faroe Islands",
	"FRA": "France",
	"GAB": "Gabon",
	"GBR": "United Kingdom",
	"GRD": "Grenada",
	"GEO": "Georgia",
	"GUF": "French Guiana",
	"GGY": "Guernsey",
	"GHA": "Ghana",
	"GIB": "Gibraltar",
	"GRL": "Greenland",
	"GMB": "Gambia",
	"GIN": "Guinea",
	"GLP": "Guadeloupe",
	"GNQ": "Equatorial Guinea",
	"GRC": "Greece",
	"SGS": "South Georgia and South Sandwich Islands",
	"GTM": "Guatemala",
	"GUM": "Guam",
	"GNB": "Guinea-Bissau",
	"GUY": "Guyana",
	"HKG": "Hong Kong",
	"HMD": "Heard Island and McDonald Islands",
	"HND": "Honduras",
	"HRV": "Croatia",
	"HTI": "Haiti",
	"HUN": "Hungary",
	"IDN": "Indonesia",
	"IRL": "Ireland",
	"ISR": "Israel",
	"IMN": "Isle of Man",
	"IND": "India",
	"IOT": "British Indian Ocean Territory",
	"IRQ": "Iraq",
	"IRN": "Iran",
	"ISL": "Iceland",
	"ITA": "Italy",
	"JEY": "Jersey",
	"JAM": "Jamaica",
	"JOR": "Jordan",
	"JPN": "Japan",
	"KEN": "Kenya",
	"KGZ": "Kyrgyzstan",
	"KHM": "Cambodia",
	"KIR": "Kiribati",
	"COM": "Comoros",
	"KNA": "St Kitts and Nevis",
	"PRK": "North Korea",
	"KOR": "South Korea",
	"KWT": "Kuwait",
	"CYM": "Cayman Islands",
	"KAZ": "Kazakhstan",
	"LAO": "Laos",
	"LBN": "Lebanon",
	"LCA": "Saint Lucia",
	"LIE": "Liechtenstein",
	"LKA": "Sri Lanka",
	"LBR": "Liberia",
	"LSO": "Lesotho",
	"LTU": "Lithuania",
	"LUX": "Luxembourg",
	"LVA": "Latvia",
	"LBY": "Libya",
	"MAR": "Morocco",
	"MCO": "Monaco",
	"MDA": "Moldova",
	"MNE": "Montenegro",
	"MAF": "Saint Martin",
	"MDG": "Madagascar",
	"MHL": "Marshall Islands",
	"MKD": "North Macedonia",
	"MLI": "Mali",
	"MMR": "Myanmar",
	"MNG": "Mongolia",
	"MAC": "Macao",
	"MNP": "Northern Mariana Islands",
	"MTQ": "Martinique",
	"MRT": "Mauritania",
	"MSR": "Montserrat",
	"MLT": "Malta",
	"MUS": "Mauritius",
	"MDV": "Maldives",
	"MWI": "Malawi",
	"MEX": "Mexico",
	"MYS": "Malaysia",
	"MOZ": "Mozambique",
	"NAM": "Namibia",
	"NCL": "New Caledonia",
	"NER": "Niger",
	"NFK": "Norfolk Island",
	"NGA": "Nigeria",
	"NIC": "Nicaragua",
	"NLD": "Netherlands",
	"NOR": "Norway",
	"NPL": "Nepal",
	"NRU": "Nauru",
	"NIU": "Niue",
	"NZL": "New Zealand",
	"OMN": "Oman",
	"PAN": "Panama",
	"PER": "Peru",
	"PYF": "French Polynesia",
	"PNG": "Papua New Guinea",
	"PHL": "Philippines",
	"PAK": "Pakistan",
	"POL": "Poland",
	"SPM": "Saint Pierre and Miquelon",
	"PCN": "Pitcairn Islands",
	"PRI": "Puerto Rico",
	"PSE": "Palestine",
	"PRT": "Portugal",
	"PLW": "Palau",
	"PRY": "Paraguay",
	"QAT": "Qatar",
	"REU": "Réunion",
	"ROU": "Romania",
	"SRB": "Serbia",
	"RUS": "Russia",
	"RWA": "Rwanda",
	"SAU": "Saudi Arabia",
	"SLB": "Solomon Islands",
	"SYC": "Seychelles",
	"SDN": "Sudan",
	"SWE": "Sweden",
	"SGP": "Singapore",
	"SHN": "Saint Helena",
	"SVN": "Slovenia",
	"SJM": "Svalbard and Jan Mayen",
	"SVK": "Slovakia",
	"SLE": "Sierra Leone",
	"SMR": "San Marino",
	"SEN": "Senegal",
	"SOM": "Somalia",
	"SUR": "Suriname",
	"SSD": "South Sudan",
	"STP": "São Tomé and Príncipe",
	"SLV": "El Salvador",
	"SXM": "Sint Maarten",
	"SYR": "Syria",
	"SWZ": "Eswatini",
	"TCA": "Turks and Caicos Islands",
	"TCD": "Chad",
	"ATF": "French Southern Territories",
	"TGO": "Togo",
	"THA": "Thailand",
	"TJK": "Tajikistan",
	"TKL": "Tokelau",
	"TLS": "Timor-Leste",
	"TKM": "Turkmenistan",
	"TUN": "Tunisia",
	"TON": "Tonga",
	"TUR": "Turkey",
	"TTO": "Trinidad and Tobago",
	"TUV": "Tuvalu",
	"TWN": "Taiwan",
	"TZA": "Tanzania",
	"UKR": "Ukraine",
	"UGA": "Uganda",
	"UMI": "U.S. Minor Outlying Islands",
	"USA": "United States",
	"URY": "Uruguay",
	"UZB": "Uzbekistan",
	"VAT": "Vatican City",
	"VCT": "St Vincent and Grenadines",
	"VEN": "Venezuela",
	"VGB": "British Virgin Islands",
	"VIR": "U.S. Virgin Islands",
	"VNM": "Vietnam",
	"VUT": "Vanuatu",
	"WLF": "Wallis and Futuna",
	"WSM": "Samoa",
	"XKX": "Kosovo",
	"YEM": "Yemen",
	"MYT": "Mayotte",
	"ZAF": "South Africa",
	"ZMB": "Zambia",
	"ZWE": "Zimbabwe"
}

