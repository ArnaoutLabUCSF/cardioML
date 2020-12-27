import sys
import json
import logging
import time
import configparser
import urllib
import datetime
import os
import random
from collections import OrderedDict
import requests
import lxml.html as lh
from pathlib import Path
import traceback
import re
import arxiv
import helpers
from database import Database
from helpers import Api
from helpers import Downloader

class Articles:
    def run(self):
        self.initialize()

        # go through each site
        for item in self.sites:
            self.doItem(item)
            self.onItemIndex += 1

        self.cleanUp()

    def doItem(self, item):
        inputType = 'search terms'

        if self.options.get('useIdLists', ''):
            inputType = 'ID list'

        self.onKeywordIndex = 0
        self.keywords = self.readInputFile(item, inputType)

        for keyword in self.keywords:
            self.showStatus(item, keyword)
        
            # already done?
            if self.isDone(item, keyword):
                self.onKeywordIndex += 1
                continue

            try:
                # do the search and download the results
                self.lookUpItem(item, keyword)
                self.markDone(item, keyword)
            except Exception as e:
                # if something goes wrong, we just go to next keyword
                logging.error(f'Skipping. Something went wrong.')
                logging.debug(traceback.format_exc())                
                logging.error(e)

            self.onKeywordIndex += 1

    def lookUpItem(self, site, keyword):
        siteName = helpers.getDomainName(site.get('url', ''))

        self.totalResults = 0
        
        articles = []

        # use pubmed's api
        if siteName == 'nih.gov':
            articles = self.nihSearch(site, keyword)
        # use arxiv's api
        elif siteName == 'arxiv.org':
            articles = self.arxivSearch(site, keyword)
        # get the website and parse it
        else:
            siteData = {}
            
            keywordWithPlusSigns = urllib.parse.quote_plus(keyword);
            keywordWithPlusSigns = keywordWithPlusSigns.replace('%20', '+')
            
            if siteName == 'biorxiv.org':
                siteData = {
                    'url': f'https://www.biorxiv.org/search/{keywordWithPlusSigns}%20numresults%3A75%20sort%3Arelevance-rank',
                    'resultsXpath': "//a[@class = 'highwire-cite-linked-title']",
                    'totalResultsXpath': "//*[@id = 'search-summary-wrapper']",
                    'titleXpath': "./span[@class = 'highwire-cite-title']",
                    'dateSubmittedXpath': "//div[@class = 'pane-content' and contains(., 'Posted')]",
                    'urlPrefix': 'https://www.biorxiv.org',
                    'afterFirstPageSuffix': '?page={}',
                    'abstractXpath' : "//*[@id = 'abstract-1']//*[@id = 'p-2']",
                    'titleInDetailsPageXpath' : "//*[@id = 'page-title']"
                }
            elif siteName == 'medrxiv.org':
                siteData = {
                    'url': f'https://www.medrxiv.org/search/{keywordWithPlusSigns}%20numresults%3A75%20sort%3Arelevance-rank',
                    'resultsXpath': "//a[@class = 'highwire-cite-linked-title']",
                    'totalResultsXpath': "//*[@id = 'search-summary-wrapper']",
                    'titleXpath': "./span[@class = 'highwire-cite-title']",
                    'dateSubmittedXpath': "//div[@class = 'pane-content' and contains(., 'Posted')]",
                    'urlPrefix': 'https://www.medrxiv.org',
                    'afterFirstPageSuffix': '?page={}',
                    'abstractXpath' : "//*[@id = 'abstract-1']//*[@id = 'p-2']",
                    'titleInDetailsPageXpath' : "//*[@id = 'page-title']"
                }

            articles = self.genericSearch(site, keyword, siteData)

        i = 0
        
        # download all the pdf url's we found
        for article in articles:
            self.outputResult(site, keyword, i + 1, article)

            i += 1

    def showStatus(self, item, keyword):
        siteName = helpers.getDomainName(item.get('url', ''))

        logging.info(f'Site {self.onItemIndex + 1} of {len(self.sites)}: {siteName}. Keyword {self.onKeywordIndex + 1} of {len(self.keywords)}: {keyword}.')

    # have enough results?
    def shouldStopForThisKeyword(self, index, log=True):
        result = False
        
        if index >= self.options['maximumResultsPerKeyword']:
            maximum = self.options['maximumResultsPerKeyword']
            
            if log:
                logging.info(f'Stopping. Reached the maximum of {maximum} results for this keyword.')
            
            result = True

        return result

    def nihSearch(self, site, keyword):
        results = []

        api = Api('http://eutils.ncbi.nlm.nih.gov')

        for i in range(0, 1000):
            pageResults = self.getNihPage(site, keyword, api, i, results, len(results))

            if not pageResults:
                logging.debug('Reached end of search results')
                break

            results += pageResults

            # have enough results?
            if self.shouldStopForThisKeyword(len(results)):
                break

            i += 1
        
        return results

    def getGenericSearchPage(self, site, keyword, siteData, pageIndex, existingResults, resultCount):
        results = []
        ids = []

        logging.info(f'Getting page {pageIndex + 1}')

        suffix = ''

        if pageIndex > 0:
            suffix = siteData.get('afterFirstPageSuffix', '')
            suffix = suffix.format(pageIndex)

        elements = []
        total = 0

        if not self.options['useIdLists']:
            page = self.downloader.get(siteData['url'] + suffix)

            elements = self.downloader.getXpath(page, siteData['resultsXpath'])
        
            total = self.downloader.getXpath(page, siteData['totalResultsXpath'], True)
            total = helpers.numbersOnly(total)
        else:
            class Element:
                def __init__(self, keyword):
                    self.attrib = {
                        'href': '/content/10.1101/' + keyword
                    }

            element = Element(keyword)
            elements = [element]
            total = 1

        if not self.totalResults and total:
            self.totalResults = int(total)

            self.showResultCount()

            # log the search now because the download might fail
            self.logToCsvFiles(site, keyword, -1, [], '', False, True, False)

        urls = []
        i = resultCount
        
        # get information about each item
        for element in elements:
            try:
                if self.shouldStopForThisKeyword(i, False):
                    break

                i += 1
            
                url = element.attrib['href']
                url = siteData['urlPrefix'] + url

                # avoids duplicates
                if url in urls:
                    continue

                urls.append(url)

                pdfUrl = url + '.full.pdf'
                    
                articleId = self.getLastAfterSplit(url, '/')

                title = ''
                
                if not self.options['useIdLists']:
                    title = self.downloader.getXpathInElement(element, siteData['titleXpath'])

                information = self.getInformationFromDetailsPage(siteData, url)

                if not title:
                    title = information.get('title', '')

                abstract = information.get('abstract', '')

                shortTitle = title

                if len(shortTitle) > 50:
                    shortTitle = shortTitle[0:50] + '...'

                logging.info(f'Results: {len(urls)}. Url: {url}. Title: {shortTitle}.')
                
                # this allows us to know when we reached the final page
                if self.isInArticleList(existingResults, articleId):
                    continue

                result = [
                    articleId,
                    pdfUrl,
                    title,
                    information.get('dateSubmitted'),
                    abstract,
                    information.get('allAuthors', ''),
                    information.get('allLocations', ''),
                    information.get('firstAuthor', ''),
                    information.get('firstAuthorLocation', ''),
                    information.get('lastAuthor', ''),
                    information.get('lastAuthorLocation', ''),
                    information.get('citations', '')              
                ]
                
                results.append(result)
            except Exception as e:
                # if something goes wrong, we just go to next keyword
                logging.error(f'Skipping. Something went wrong.')
                logging.debug(traceback.format_exc())                
                logging.error(e)

        return results

    def getInformationFromDetailsPage(self, siteData, url):
        page = self.downloader.get(url)

        title = self.downloader.getXpath(page, siteData['titleInDetailsPageXpath'], True)
        
        dateSubmitted = self.downloader.getXpath(page, siteData['dateSubmittedXpath'], True)
        # it starts with a non-breaking space
        dateSubmitted = helpers.findBetween(dateSubmitted, '\xa0', '.')
        
        abstract = self.downloader.getXpath(page, siteData['abstractXpath'], True)

        if dateSubmitted:
            dateSubmitted = self.changeDateFormat(dateSubmitted, '%B %d, %Y')

        import lxml.html as lh
        document = lh.fromstring(page)

        allAuthors = []
        allLocations = []
        firstAuthor = ''
        firstAuthorLocation = ''
        lastAuthor = ''
        lastAuthorLocation = ''

        authorXpath = "//*[contains(@id, 'hw-article-author-popups-')]/div[contains(@class, 'author-tooltip-')]"

        elements = document.xpath(authorXpath)

        for i, element in enumerate(elements):
            name = self.downloader.getXpathInElement(element, ".//div[@class = 'author-tooltip-name']", False)

            name = name.strip()

            if not name:
                continue

            allAuthors.append(name)

            if not firstAuthor:
                firstAuthor = name
            # only if the article has a last author
            elif i > 0 and i == len(elements) - 1 and not lastAuthor:
                lastAuthor = name

            affiliations = element.xpath(".//span[@class = 'nlm-aff']")
            
            for affiliation in affiliations:
                location = affiliation.text_content()

                location = location.strip()

                if not location:
                    continue

                if i == 0 and not firstAuthorLocation:
                    firstAuthorLocation = location
                # only if the article has a last author
                elif i > 0 and i == len(elements) - 1 and not lastAuthorLocation:
                    lastAuthorLocation = location

                # avoid duplicates
                if not location in allLocations:
                    allLocations.append(location)

        result = {
            'title': title,
            'dateSubmitted': dateSubmitted,
            'abstract': abstract,
            'allAuthors': '; '.join(allAuthors),
            'allLocations': ' | '.join(allLocations),
            'firstAuthor': firstAuthor,
            'firstAuthorLocation':firstAuthorLocation,
            'lastAuthor': lastAuthor,
            'lastAuthorLocation': lastAuthorLocation,
            'citations': ''
        }

        return result

    def changeDateFormat(self, dateString, sourceFormat):
        date = datetime.datetime.strptime(dateString, sourceFormat)

        return date.strftime('%Y-%m-%d')

    def showResultCount(self):
        maximumResults = self.options['maximumResultsPerKeyword']

        logging.info(f'Total number of results available: {self.totalResults}. Number of desired results: {maximumResults}.' )


    def isInArticleList(self, articleList, articleId):
        result = False

        for article in articleList:
            if len(article) >= 2 and article[0] == articleId:
                result = True
                break

        return result

    def getNihPage(self, site, keyword, api, pageIndex, existingResults, resultCount):
        results = []

        resultsPerPage = 1000
        start = pageIndex * resultsPerPage
        response = ''

        if not self.options.get('useIdLists', ''):
            logging.info(f'Getting page {pageIndex + 1}')
    
            response = api.get(f'/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retstart={start}&retmax={resultsPerPage}&term={keyword}')

            if not response:
                logging.error('No response')
                return []

            if not self.totalResults:
                self.totalResults = response['esearchresult']['count']
            
                self.showResultCount()
            
                # log the search now because the download might fail
                self.logToCsvFiles(site, keyword, -1, [], '', False, True, False)
        else:
            if pageIndex == 0:
                response = {
                    'esearchresult': {
                        'idlist': [keyword]
                    }
                }
            else:
                return []

        i = resultCount
        
        for item in response['esearchresult']['idlist']:
            if self.shouldStopForThisKeyword(i, False):
                break

            # avoid duplicates
            if self.isInArticleList(existingResults, item):
                continue

            i += 1

            try:
                summaryResponse = api.get(f'/entrez/eutils/esummary.fcgi?db=pubmed&id={item}&retmode=json')

                title = ''
                abstract = ''

                if 'result' in summaryResponse and item in summaryResponse['result']:
                    articleSummary = summaryResponse['result'][item]
                    
                    title = articleSummary.get('title', '')
                    
                    shortTitle = title

                    if len(shortTitle) > 50:
                        shortTitle = shortTitle[0:50] + '...'

                    dateSubmitted = articleSummary.get('sortpubdate', '')
                    dateSubmitted = helpers.findBetween(dateSubmitted, '', ' ')
                    dateSubmitted = dateSubmitted.replace('/', '-')

                    details = self.getNihDetails(api, item, articleSummary)
                   
                    abstract = details.get('abstract', '')

                    logging.info(f'Results: {i}. Id: {item}. Title: {shortTitle}.')

                    # write these results to a separate csv
                    self.logNihResultToCsvFile(site, keyword, articleSummary, details)

                pdfUrl = ''
            except Exception as e:
                # if something goes wrong, we just go to next keyword
                logging.error(f'Skipping {item}. Something went wrong.')
                logging.debug(traceback.format_exc())                
                logging.error(e)
                continue
            
            result = [item, pdfUrl, title, dateSubmitted, abstract]

            fields = ['allAuthors', 'allLocations', 'firstAuthor', 'firstAuthorLocation', 'lastAuthor', 'lastAuthorLocation', 'citations']

            for field in fields:
                result.append(details.get(field, ''))
            
            results.append(result)

        return results

    def getNihDetails(self, api, articleId, article):
        import xmltodict
        
        response = api.get(f'/entrez/eutils/efetch.fcgi?db=pubmed&id={articleId}&retmode=xml')
        
        details = xmltodict.parse(response)

        referenceList = helpers.getNested(details, ['PubmedArticleSet', 'PubmedArticle', 'PubmedData', 'ReferenceList'])

        # sometimes it's in a different format
        if not isinstance(referenceList, list):
            referenceList = helpers.getNested(details, ['PubmedArticleSet', 'PubmedArticle', 'PubmedData', 'ReferenceList', 'Reference'])

        details = helpers.getNested(details, ['PubmedArticleSet', 'PubmedArticle', 'MedlineCitation', 'Article'])

        details['ReferenceList'] = referenceList

        allAuthors = []

        for author in article.get('authors', ''):
            allAuthors.append(author.get('name', ''))

        allAuthors = '; '.join(allAuthors)
        allLocations = []
        firstAuthor = article.get('sortfirstauthor', '')
        firstAuthorLocation = ''
        lastAuthor = article.get('lastauthor', '')
        lastAuthorLocation = ''
        citations = []

        authorList = helpers.getNested(details, ['AuthorList', 'Author'])

        # sometimes it returns a single author rather than a list
        if not isinstance(authorList, list):
            authorList = [authorList]

        for i, author in enumerate(authorList):
            # an author can have multiple affiliations
            # it's a dictionary for one result. list for more than one.
            affiliations = author.get('AffiliationInfo', [])

            if not isinstance(affiliations, list):
                affiliations = [affiliations]
            
            for affiliation in affiliations:
                location = affiliation.get('Affiliation', '')

                if not location:
                    continue

                if i == 0 and not firstAuthorLocation:
                    firstAuthorLocation = location
                # only if the article has a last author
                elif article.get('lastauthor', '') and i > 0 and i == len(authorList) - 1 and not lastAuthorLocation:
                    lastAuthorLocation = location

                # avoid duplicates
                if not location in allLocations:
                    allLocations.append(location)

        allLocations = ' | '.join(allLocations)

        for reference in details.get('ReferenceList', ''):
            if isinstance(reference, str):
                continue

            if reference.get('Reference', ''):
                reference = reference.get('Reference', '')

            string = reference.get('Citation', '')

            id = helpers.getNested(reference, ['ArticleIdList', 'ArticleId', '#text'])

            if id:
                string += f' (PMID: {id})'

            citations.append(string)

        citations = ' | '.join(citations)

        abstractSections = []

        # sometimes it returns a single string rather than a list of strings
        oneSection = helpers.getNested(details, ['Abstract', 'AbstractText', '#text'])

        if not oneSection:
            # sometimes it's like this
            oneSection = helpers.getNested(details, ['Abstract', 'AbstractText'])

        if oneSection and isinstance(oneSection, str):
            abstractSections = [oneSection]
        else:
            for item in helpers.getNested(details, ['Abstract', 'AbstractText']):
                abstractSection = item.get('@Label', '') + ': ' + item.get('#text', '')
                abstractSections.append(abstractSection)

        abstract = '\n\n'.join(abstractSections)

        result = {
            'abstract' : abstract,
            'allAuthors': allAuthors,
            'allLocations': allLocations,
            'firstAuthor': firstAuthor,
            'firstAuthorLocation':firstAuthorLocation,
            'lastAuthor': lastAuthor,
            'lastAuthorLocation': lastAuthorLocation,
            'citations': citations
        }

        return result
    
    def arxivSearch(self, site, keyword):
        results = []

        maximumResults = self.options['maximumResultsPerKeyword']

        if maximumResults == -1:
            maximumResults = None

        items = arxiv.query(query=keyword,
                    id_list=[],
                    max_results=maximumResults,
                    start = 0,
                    sort_by="relevance",
                    sort_order="descending",
                    prune=True,
                    iterative=False,
                    max_chunk_results=1000)

        ids = []

        for item in items:
            id = item.get('id', '')
            id = self.getLastAfterSplit(id, '/')

            # avoids duplicates
            if id in ids:
                continue

            ids.append(id)

            pdfUrl = item.get('pdf_url', '')

            if not pdfUrl:
                siteName = helpers.getDomainName(site.get('url', ''))
                message = f'No pdf file found on {siteName} for {id}'
                logging.error(message)
                pdfUrl = f'Error: {message}'

            title = item.get('title', '')
            title = title.replace('\n', ' ')
            title = self.squeezeWhitespace(title)

            dateSubmitted = item.get('published', '')

            dateSubmitted = helpers.findBetween(dateSubmitted, '', 'T')

            shortTitle = title

            if len(shortTitle) > 50:
                shortTitle = shortTitle[0:50] + '...'

            abstract = item.get('summary', '')

            allAuthors = '; '.join(item.get('authors', ''))
            allLocations = ''
            firstAuthor = self.getFirst(item.get('authors', ''))
            firstAuthorLocation = ''
            lastAuthor = self.getLast(item.get('authors', ''))
            lastAuthorLocation = ''
            citations = ''

            result = [id, pdfUrl, title, dateSubmitted, abstract, allAuthors, allLocations, firstAuthor, firstAuthorLocation, lastAuthor, lastAuthorLocation, citations]
            
            results.append(result)

            logging.info(f'Results: {len(results)}. Id: {id}. Title: {shortTitle}.')

        self.totalResults = len(results)

        self.showResultCount()

        # log the search now because the download might fail
        self.logToCsvFiles(site, keyword, -1, [], '', False, True, False)

        return results

    def getFirst(self, array):
        if isinstance(array, list) and len(array) > 0:
            return array[0]

        return ''

    def getLast(self, array):
        if isinstance(array, list) and len(array) > 0:
            return array[-1]

        return ''

    def getLastAfterSplit(self, s, splitter):
        result = ''

        fields = s.split(splitter)

        if len(fields) > 0:
            result = fields[-1]

        return result

    def getArticleId(self, site, pdfUrl):
        return getLastAfterSplit(pdfUrl, '/')

    def genericSearch(self, site, keyword, siteData):
        results = []

        for i in range(0, 1000):
            pageResults = self.getGenericSearchPage(site, keyword, siteData, i, results, len(results))

            if not pageResults:
                logging.debug('Reached end of search results')
                break

            results += pageResults

            # have enough results?
            if self.shouldStopForThisKeyword(len(results)):
                break

            i += 1
        
        return results

    def outputResult(self, site, keyword, resultNumber, article):
        siteName = helpers.getDomainName(site.get('url', ''))

        articleId = article[0]
        pdfUrl = article[1]

        downloaded = 'Not downloaded'
        outputFileName = ''
        
        # log to the csv file anyway
        self.logToCsvFiles(site, keyword, resultNumber, article, outputFileName, downloaded, False, True)

        self.waitBetween()

    # log to search log and/or pdf log
    def logToCsvFiles(self, site, keyword, resultNumber, article, outputFileName, downloaded, searchLog, pdfLog):
        helpers.makeDirectory(self.options['outputDirectory'])
        
        searchLogFileName = os.path.join(self.options['outputDirectory'], 'output_searchlog.csv')
        pdfLogFileName = os.path.join(self.options['outputDirectory'], 'all_results.csv')
        
        if searchLog and not os.path.exists(searchLogFileName):
            helpers.toFile('Date-Time,Search terms,Websites,Number of papers,Requested maximumResultsPerKeyword', searchLogFileName)

        if pdfLog and not os.path.exists(pdfLogFileName):
            helpers.toFile('Datetime, Search terms, Website, Result number, Total results requested, ID number, Title, Date Submitted, Abstract, Downloaded?, FileNamePath, all_authors, all_locations, first_author, firstauthor_location, lastauthor, last_author_location, citations', pdfLogFileName)

        now = datetime.datetime.now().strftime('%m%d%y-%H%M%S')

        siteName = site.get('name', '')

        articleId = ''
        title = ''
        dateSubmitted = ''
        abstract = ''

        if len(article) >= 4:
            articleId = article[0]
            title = article[2]
            dateSubmitted = article[3]

        if len(article) >= 5:
            abstract = article[4]

        searchLogLine = [now, keyword, siteName, self.totalResults, self.options['maximumResultsPerKeyword']]
        pdfLogLine = [now, keyword, siteName, resultNumber, self.options['maximumResultsPerKeyword'], articleId, title, dateSubmitted, abstract, downloaded, outputFileName]

        if searchLog:
            self.appendCsvFile(searchLogLine, searchLogFileName)

        if pdfLog:
            if len(article) >= 6:
                pdfLogLine += article[5:]

            self.appendCsvFile(pdfLogLine, pdfLogFileName)

    # writes article details to a csv file
    def logNihResultToCsvFile(self, site, keyword, article, articleDetails):
        name = site.get('name', '').lower()
        
        csvFileName = os.path.join(self.options['outputDirectory'], f'{name}_results.csv')
        
        if not os.path.exists(csvFileName):
            helpers.makeDirectory(os.path.dirname(csvFileName))
            helpers.toFile('DateTime,Keyword,Title,Date_Submitted,URL,Abstract,Description,Details,ShortDetails,Resource,Type,Identifiers,Db,EntrezUID,Properties,all_authors,all_locations,first_author,firstauthor_location,lastauthor,last_author_location,citations', csvFileName)

        siteName = site.get('name', '')

        articleId = article.get('uid', '')

        dateSubmitted = article.get('sortpubdate', '')
        dateSubmitted = helpers.findBetween(dateSubmitted, '', ' ')
        dateSubmitted = dateSubmitted.replace('/', '-')

        properties = 'create date: ' + helpers.findBetween(article.get('sortpubdate', ''), '', ' ') + ' | first author: ' + article.get('sortfirstauthor', '')

        description = ''

        authors = []
        for author in article.get('authors', ''):
            authors.append(author.get('name', ''))

        description = ', '.join(authors) + '.'

        publicationTypes = ', '.join(article.get('pubtype', []))

        details = article.get('fulljournalname', '') + '. ' + article.get('elocationid', '') + '. ' + publicationTypes + '.'
        
        line = [
            datetime.datetime.now().strftime('%m%d%y-%H%M%S'),
            keyword,
            article.get('title', ''),
            dateSubmitted,
            f'/pubmed/{articleId}',
            articleDetails.get('abstract', ''),
            description,
            details,
            article.get('fulljournalname', '') + '. ' + helpers.findBetween(article.get('sortpubdate', ''), '', '/'),
            'PubMed',
            publicationTypes,
            f'PMID:{articleId}',
            'pubmed',
            articleId,
            properties,
            articleDetails.get('allAuthors', ''),
            articleDetails.get('allLocations', ''),
            article.get('sortfirstauthor', ''),
            articleDetails.get('firstAuthorLocation', ''),
            article.get('lastauthor', ''),
            articleDetails.get('lastAuthorLocation', ''),
            articleDetails.get('citations', '')
        ]
        
        self.appendCsvFile(line, csvFileName)

    def appendCsvFile(self, line, fileName):
        import csv
        with open(fileName, "a", newline='\n', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(line)

    def handleCaptcha(self, siteName, outputFileName):
        result = False

        if siteName != 'nih.gov':
            return result

        try:
            statinfo = os.stat(outputFileName)
        
            if statinfo.st_size < 1000 * 1000:
                file = helpers.getBinaryFile(outputFileName)

                if not file.startswith(b'%PDF'):
                    logging.error(f'Can\'t download this file. There is a captcha.')

                    # delete the file
                    if os.path.exists(outputFileName):
                        os.remove(outputFileName)
                    result = True
        except Exception as e:
            logging.error(e)

        return result

    def existsInDirectory(self, fileName):
        result = False;

        if self.options['directoryToCheckForDuplicates'] != 1:
            return result
        
        for file in helpers.listFiles(self.options['outputDirectory'], False):
            if helpers.fileNameOnly(file, True) == fileName:
                outputDirectory = self.options['outputDirectory']
                logging.info(f'Skipping. Output file already exists in {outputDirectory}.')
                result = True
                break

        return result

    def isDone(self, site, keyword):
        result = False;

        siteName = helpers.getDomainName(site.get('url', ''))

        keyword = keyword.replace("'", "''")

        directory = self.options['outputDirectory']

        siteName = self.database.getFirst('history', 'siteName', f"siteName= '{siteName}' and keyword = '{keyword}' and directory = '{directory}'", '', '')

        if siteName:
            logging.info(f'Skipping. Already done this item.')
            result = True

        return result

    # so we know not to repeat this site/keyword too soon
    def markDone(self, site, keyword):
        siteName = helpers.getDomainName(site.get('url', ''))

        keyword = keyword.replace("'", "''")

        item = {
            'siteName': siteName,
            'keyword': keyword,
            'directory': self.options['outputDirectory'],
            'gmDate': str(datetime.datetime.utcnow())
        }

        logging.debug(f'Inserting into database')
        logging.debug(item)
            
        self.database.insert('history', item)

    def waitBetween(self):
        secondsBetweenItems = self.options['secondsBetweenItems']

        if not secondsBetweenItems:
            return

        logging.info(f'Waiting {secondsBetweenItems} seconds')

        time.sleep(secondsBetweenItems)

    def readInputFile(self, site, inputType):
        results = []

        fileName = ''

        # the command line parameter takes priority
        if self.options['inputKeywordsFile']:
            fileName = self.options['inputKeywordsFile']
        else:
            siteName = site.get('name').lower()    
            
            if inputType == 'search terms':
                fileName = self.keywordsFiles.get(siteName, '')
            else:
                fileName = self.idListFiles.get(siteName, '')

        if not fileName:
            logging.error(f'No {inputType} file specified for {siteName}.')
            input("Press enter to continue...")
        
        logging.info(f'Using {inputType} file: {fileName}')

        file = helpers.getFile(fileName)

        for line in file.splitlines():
            results.append(line)

        if not results:
            logging.error('No search terms or ID\'s found')
            input("Press enter to continue...")

        return results

    def setOptionFromParameter(self, optionName, parameterName):
        if not parameterName in sys.argv:
            return

        self.options[optionName] = helpers.getArgument(parameterName, False)

    def removeOldEntries(self):
        maximumDaysToKeepItems = self.options['maximumDaysToKeepItems']
        
        minimumDate = helpers.getDateStringSecondsAgo(maximumDaysToKeepItems * 24 * 60 * 60, True)
        
        logging.debug(f'Deleting entries older than {maximumDaysToKeepItems} days')
        self.database.execute(f"delete from history where gmDate < '{minimumDate}'")

    def squeezeWhitespace(self, s):
        return re.sub(r'\s\s+', " ", s)

    def cleanUp(self):
        self.database.close()

        logging.info('Done')

    def initialize(self):
        suffix = helpers.getArgument('-w', False)
        
        if suffix:
            suffix = '-' + helpers.fileNameOnly(suffix, False)

        helpers.setUpLogging(suffix)

        logging.info('Starting\n')

        self.onItemIndex = 0
        self.onKeywordIndex = 0

        # to store the time we finished given sites/keyword combinations
        self.database = Database('database.sqlite')
        self.database.execute('create table if not exists history ( siteName text, keyword text, directory text, gmDate text, primary key(siteName, keyword, directory) )')

        self.downloader = Downloader()
        self.dateStarted = datetime.datetime.now().strftime('%m%d%y')
        
        outputDirectory = os.path.join(str(Path.home()), 'Desktop', f'WebSearch_{self.dateStarted}')

        # set default options
        self.options = {
            'inputWebsitesFile': 'input_websites.txt',
            'inputKeywordsFile': '',
            'outputDirectory': outputDirectory,
            'secondsBetweenItems': 0,
            'maximumDaysToKeepItems': 90,
            'maximumResultsPerKeyword': 25000,
            'directoryToCheckForDuplicates': '',
            'useIdLists': 0
        }

        self.keywordsFiles = {}
        self.idListFiles = {}
        
        # read the options file
        helpers.setOptions('options.ini', self.options)
        helpers.setOptions('options.ini', self.keywordsFiles, 'search terms')
        helpers.setOptions('options.ini', self.idListFiles, 'id lists')

        # read command line parameters
        self.setOptionFromParameter('inputWebsitesFile', '-w')
        self.setOptionFromParameter('inputKeywordsFile', '-s')
        self.setOptionFromParameter('outputDirectory', '-d')

        if '-i' in sys.argv:
            self.options['maximumResultsPerKeyword'] = 1
            logging.info('Downloading by ID list')
            self.options['useIdLists'] = 1

        # read websites file
        file = helpers.getFile(self.options['inputWebsitesFile'])
        self.sites = []

        for item in file.splitlines():
            name = helpers.findBetween(item, '', ' ')
            url = helpers.findBetween(item, ' ', '')

            site = {
                'name': name,
                'url': url
            }

            self.sites.append(site)

        self.removeOldEntries()

articles = Articles()
articles.run()