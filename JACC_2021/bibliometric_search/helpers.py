import sys
import io
import logging
import os.path
import csv
import subprocess
import random
import time
import configparser
import datetime
import json
from logging.handlers import RotatingFileHandler
from collections import OrderedDict

def getFile(fileName, encoding=None):
    if not os.path.isfile(fileName):
        return ""

    f = open(fileName, "r", encoding='utf-8')
    
    return f.read()

def getBinaryFile(fileName):
    if not os.path.isfile(fileName):
        return ""

    f = open(fileName, "rb")
    
    return f.read()


def getLines(fileName):
    if not os.path.isfile(fileName):
        return []

    with open(fileName) as f:
        return f.readlines()


def toFile(s, fileName):
    with io.open(fileName, "w", encoding="utf-8") as text_file:
        print(s, file=text_file)

def toBinaryFile(s, fileName):
    with io.open(fileName, "wb") as file:
        file.write(s)

def appendToFile(s, fileName):
    with io.open(fileName, "a", encoding="utf-8") as text_file:
        print(s, file=text_file)


def numbersOnly(s):
    return ''.join(filter(lambda x: x.isdigit(), s))

def fixedDecimals(n, numberOfDecimalPlaces):
    result = ''

    try:
        formatString = f'{{:.{numberOfDecimalPlaces}f}}'

        result = formatString.format(n)
    except Exception as e:
        logging.error(e)

    return result

def findBetween(s, first, last):
    try:
        start = s.index(first) + len(first)
    except ValueError:
        start = 0

    try:
        if not last:
            end = len(s)
        else:
            end = s.index(last, start)
    except ValueError:
        end = len(s)

    return s[start:end]

def getNested(j, keys):
    try:
        element = j

        i = 0

        for key in keys:
            if not key in element:
                break
            
            element = element[key]

            if i == len(keys) - 1:
                return element

            i += 1
    except Exception as e:
        logging.error(e)
        return ""

    return ""

def stringToFloatingPoint(s):
    result = 0.0

    temporary = ""

    for c in s:
        if c.isdigit() or c == ".":
            temporary += c

    try:
        result = float(temporary)
    except:
        result = 0.0

    return result

def getCsvFile(fileName):
    result = []

    with open('input.csv') as inputFile:
        csvReader = csv.reader(inputFile, delimiter=',')
            
        # skip the headers
        next(csvReader, None)

        for row in csvReader:
            if len(row) == 0:
                continue

            result.append(row)

    return result

def getCsvFileAsDictionary(fileName):
    result = []

    with open('input.csv') as inputFile:
        csvReader = csv.DictReader(inputFile, delimiter=',')
            
        for row in csvReader:
            if len(row) == 0:
                continue

            result.append(row)

    return result

def makeDirectory(directoryName):
    import pathlib
    pathlib.Path(directoryName).mkdir(parents=True, exist_ok=True)

def run(command):
    try:
        subprocess.run(command)
    except Exception as e:
        logging.error(e)

def getStandardOutput(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE)
    except Exception as e:
        logging.error(e)
        return ''
    
    return result.stdout.decode('utf-8')

def runWithInput(command, input):
    try:
        result = subprocess.run(command, input=input, encoding='ascii')
    except Exception as e:
        logging.error(e)
        return None

    return result

def getUrl(url):
    response = ''

    try:
        import requests
        response = requests.get(url)
    except Exception as e:
        logging.error(e)
        return ''
        
    return response.text

def sleep(seconds):
    time.sleep(int(seconds))

def setOptions(fileName, options, sectionName='main'):
    try:
        if '--optionsFile' in sys.argv:
            index = sys.argv.index('--optionsFile')
            if index < len(sys.argv) - 1:
                fileName = sys.argv[index + 1]

        optionsReader = configparser.ConfigParser(interpolation=None)
        optionsReader.optionxform = str 
        optionsReader.read(fileName)

        for section in optionsReader.sections():
            if sectionName and section != sectionName:
                continue

            if not sectionName:
                options[section] = {}

            for key in optionsReader[section]:
                # default value is digit?
                if isinstance(options.get(key, ''), int):
                    if not sectionName:                    
                        options[section][key] = int(optionsReader[section][key])
                    else:
                        options[key] = int(optionsReader[section][key])
                else:
                    if not sectionName:
                        options[section][key] = optionsReader[section][key]
                    else:
                        options[key] = optionsReader[section][key]
    except Exception as e:
        logging.error(e)

def getArgument(name, required):
    result = ''

    try:
        if name in sys.argv:
            index = sys.argv.index(name)
            if index < len(sys.argv) - 1:
                result = sys.argv[index + 1]

    except Exception as e:
        logging.error(e)

    if required and not result:
        logging.error(f'Parameter {name} is required')
        input("Press enter to exit...")
        exit()

    return result

def timeAgo(time=False):
    """
    Get a datetime object or a int() Epoch timestamp and return a
    pretty string like 'an hour ago', 'Yesterday', '3 months ago',
    'just now', etc
    """
    diff = 0

    now = datetime.now()
    if type(time) is float:
        diff = now - datetime.fromtimestamp(time)
    elif isinstance(time,datetime):
        diff = now - time
    elif not time:
        diff = now - now
    second_diff = diff.seconds
    day_diff = diff.days

    if day_diff < 0:
        return ''

    if day_diff == 0:
        if second_diff < 10:
            return "just now"
        if second_diff < 60:
            return str(second_diff) + " seconds ago"
        if second_diff < 120:
            return "a minute ago"
        if second_diff < 3600:
            return str(second_diff / 60) + " minutes ago"
        if second_diff < 7200:
            return "an hour ago"
        if second_diff < 86400:
            return str(second_diff / 3600) + " hours ago"
    if day_diff == 1:
        return "Yesterday"
    if day_diff < 7:
        return str(day_diff) + " days ago"
    if day_diff < 31:
        return str(day_diff / 7) + " weeks ago"
    if day_diff < 365:
        return str(day_diff / 30) + " months ago"
    return str(day_diff / 365) + " years ago"

def fileNameOnly(fileName, includeExtension):
    result = os.path.basename(fileName)

    if not includeExtension:
        result = os.path.splitext(result)[0]

    return result

def addToStartup(fileName):
    import getpass
    
    userName = getpass.getuser()

    directoryName = os.path.abspath(fileName)
    directoryName = os.path.dirname(directoryName)

    batFileName = fileNameOnly(fileName, False) + '.bat'
    
    # uses same file name twice
    startupScriptFileName = os.path.join(directoryName, batFileName)

    batPath = r'C:\Users\%s\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup' % userName
    
    with open(batPath + '\\' + batFileName, 'w+') as file:
        file.write(os.path.splitdrive(directoryName)[0] + '\n')
        file.write('cd ' + directoryName + '\n')
        file.write(r'start /min %s' % startupScriptFileName)

def setUpLogging(fileNameSuffix=''):
    logFormatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)        
    rootLogger.addHandler(consoleHandler)

    logFileName = os.path.join('logs', f'log{fileNameSuffix}.txt')

    if '--debug' in sys.argv:
        if os.path.exists(logFileName):
            # clear the file
            open(logFileName, 'w').close()
            rootLogger.setLevel(logging.DEBUG)

    makeDirectory(os.path.dirname(logFileName))

    # 2 rotating files of maximum 1 million bytes each        
    fileHandler = RotatingFileHandler(logFileName, maxBytes=1000 * 1000, backupCount=1, encoding='utf-8')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

def getDateStringSecondsAgo(secondsAgo, useGmTime):
    now = None

    if useGmTime:
        now = datetime.datetime.utcnow()
    else:
        now = datetime.datetime.now()

    result = now - datetime.timedelta(seconds=secondsAgo)
    
    return str(result)

def getDomainName(url):
    result = ''

    from urllib.parse import urlparse
    parsed_uri = urlparse(url)
    location = '{uri.netloc}'.format(uri=parsed_uri)

    fields = location.split('.')

    if len(fields) >= 2:
        result = fields[-2] + '.' + fields[-1]

    return result

class Api:
    def get(self, url):
        import requests

        result = ''

        try:
            logging.debug(f'Get {url}')

            response = requests.get(self.urlPrefix + url, headers=self.headers, proxies=self.proxies)

            if response.text[0] == '{' or response.text[0] == '[':
                result = json.loads(response.text)
            else:
                result = response.text
        except Exception as e:
            logging.error(e)

        return result

    def post(self, url, data, responseIsJson=True):
        import requests
        
        result = ''

        try:
            logging.debug(f'Post {url}')

            response = requests.post(self.urlPrefix + url, headers=self.headers, proxies=self.proxies, data=data)

            logging.debug(response)
            logging.debug(response.headers)

            if response.headers['Content-Type'] == 'application/pdf' or response.headers['content-type'] == 'application/pdf':
                result = response.content
            elif responseIsJson:
                result = json.loads(response.text)
            else:
                result = response.text
        except Exception as e:
            logging.error(e)

        return result

    def __init__(self, urlPrefix):
        self.urlPrefix = urlPrefix

        self.userAgentList = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0"
        ]

        userAgent = random.choice(self.userAgentList)

        self.headers = OrderedDict([
            ('user-agent', userAgent),
            ('accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
            ('accept-language', 'en-US,en;q=0.5'),
            ('dnt', '1'),
            ('upgrade-insecure-requests', '1'),
            ('te', 'trailers')
        ])

        self.proxies = None
        
def fileNameOnly(fileName, includeExtension):
    result = os.path.basename(fileName)

    if not includeExtension:
        result = os.path.splitext(result)[0]

    return result

def addToStartup(fileName):
    import getpass
    
    userName = getpass.getuser()

    directoryName = os.path.abspath(fileName)
    directoryName = os.path.dirname(directoryName)

    batFileName = fileNameOnly(fileName, False) + '.bat'
    
    # uses same file name twice
    startupScriptFileName = os.path.join(directoryName, batFileName)

    batPath = r'C:\Users\%s\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup' % userName
    
    with open(batPath + '\\' + batFileName, 'w+') as file:
        file.write(os.path.splitdrive(directoryName)[0] + '\n')
        file.write('cd ' + directoryName + '\n')
        file.write(r'start /min %s' % startupScriptFileName)

class Downloader:
    def get(self, url):
        import requests

        userAgent = random.choice(self.userAgentList)
        
        self.headers = OrderedDict([
            ('user-agent', userAgent),
            ('accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'),
            ('accept-language', 'en-US,en;q=0.9')
        ])

        self.proxies = None

        response = ''

        try:
            logging.debug(f'Getting {url}')
            response = requests.get(url, headers=self.headers, proxies=self.proxies)
            response.encoding = 'utf-8'
        except Exception as e:
            logging.error(e)
            return ''
        
        return response.text

    def downloadBinaryFile(self, url, destinationFileName):       
        result = False
        
        import wget
        
        logging.debug(f'Download {url} to {destinationFileName}')
        
        try:
            wget.download(url, destinationFileName)
            result = True
        except Exception as e:
            logging.error(e)
        
        return result

    def getXpath(self, page, xpath, firstOnly=False, attribute=None):
        result = []

        if firstOnly:
            result = ''
        
        try:
            import lxml.html as lh

            result = ''

            document = lh.fromstring(page)

            # get matching elements
            elements = document.xpath(xpath)

            if firstOnly:
                if len(elements) > 0:
                    if not attribute:
                        result = elements[0].text_content()
                    else:
                        result = elements[0].attrib[attribute]
            else:
                result = elements
        except Exception as e:
            logging.error(e)

        return result

    # xpath should start with "./" instead of "//"
    def getXpathInElement(self, rootElement, xpath, attribute=None):
        result = ''

        try:
            elements = rootElement.xpath(xpath)

            if len(elements) > 0:
                if not attribute:
                    result = elements[0].text_content()
                else:
                    result = elements[0].attrib[attribute]
        except Exception as e:
            logging.error(e)

        return result


    def __init__(self):
        self.userAgentList = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36'
        ]

def listFiles(directory, includeDirectories=True):
    result = []

    root = directory

    for dirname, dirnames, filenames in os.walk(root):
        # print path to all subdirectories first.
        if includeDirectories:
            for subdirname in dirnames:
                result.append(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            result.append(os.path.join(dirname, filename))
    
    return result