###########################################################################
#
## @file database.py
#
###########################################################################

import sqlite3
import logging
import time
import random

###########################################################################
#
##   A wrapper around the sqlite3 python library.
#
#    The Database class is a high-level wrapper around the sqlite3
#    library. It allows users to create a database connection and
#    write to or fetch data from the selected database. It also has
#    various utility functions such as getLast(), which retrieves
#    only the very last item in the database, toCSV(), which writes
#    entries from a database to a CSV file, and summary(), a function
#    that takes a dataset and returns only the maximum, minimum and
#    average for each column. The Database can be opened either by passing
#    on the name of the sqlite database in the constructor, or optionally
#    after constructing the database without a name first, the open()
#    method can be used. Additionally, the Database can be opened as a
#    context method, using a 'with .. as' statement. The latter takes
#    care of closing the database.
#
###########################################################################

class Database:

    #######################################################################
    #
    ## The constructor of the Database class
    #
    #  The constructor can either be passed the name of the database to open
    #  or not, it is optional. The database can also be opened manually with
    #  the open() method or as a context manager.
    #
    #  @param name Optionally, the name of the database to open.
    #
    #  @see open()
    #
    #######################################################################
    
    def __init__(self, name=None):
        
        self.conn = None
        self.cursor = None

        if name:
            self.open(name)


    #######################################################################
    #
    ## Opens a new database connection.
    #
    #  This function manually opens a new database connection. The database
    #  can also be opened in the constructor or as a context manager.
    #
    #  @param name The name of the database to open.
    #
    #  @see \__init\__()
    #
    #######################################################################
    
    def open(self,name):
        
        try:
            self.conn = sqlite3.connect(name);
            # to get column names
            self.conn.row_factory = sqlite3.Row 
            self.cursor = self.conn.cursor()

        except sqlite3.Error as e:
            print("Error connecting to database!")


    #######################################################################
    #
    ## Function to close a datbase connection.
    #
    #  The database connection needs to be closed before you exit a program,
    #  otherwise changes might be lost. You can also manage the database
    #  connection as a context manager, then the closing is done for you. If
    #  you opened the database connection with the open() method or with the
    #  constructor ( \__init\__() ), you must close the connection with this
    #  method.
    #
    #  @see open()
    #
    #  @see \__init\__()
    #
    #######################################################################
    
    def close(self):
        
        if self.conn:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()


    def __enter__(self):
        
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        
        self.close()


    #######################################################################
    #
    ## Function to fetch/query data from a database.
    #
    #  This is the main function used to query a database for data.
    #
    #  @param table The name of the database's table to query from.
    #
    #  @param columns The string of columns, comma-separated, to fetch.
    #
    #  @param limit Optionally, a limit of items to fetch.
    #
    #######################################################################

    def get(self,table,columns,where,orderBy,orderType,limit=None):
        result = []
        
        try:
            wherePart = ''
            orderByPart = ''
            limitPart = ''

            if where:
                wherePart = f' where {where}'

            if orderBy:
                orderByPart = f' order by {orderBy} {orderType}'

            if limit:
                limitPart = f' limit {limit}'

            query = f"SELECT {columns} from {table}{wherePart}{orderByPart}{limitPart};"
            
            self.executeWithRetries(query)

            rows = self.cursor.fetchall()
            for row in rows:
                result += [dict(row)]
        except Exception as e:
            logging.error(e)

        self.conn.commit()

        return result

    def getFirst(self,table,columns,where,orderBy,orderType):
        result = {}

        rows = self.get(table, columns, where, orderBy, orderType, 1)

        if len(rows) > 0:
            result = rows[0]

        return result

    #######################################################################
    #
    ## Utilty function to get the last row of data from a database.
    #
    #  @param table The database's table from which to query.
    #
    #  @param columns The columns which to query.
    #
    #######################################################################

    def getLast(self,table,columns):
        
        return self.get(table,columns,limit=1)[0]

    
    #######################################################################
    #
    ## Utility function that converts a dataset into CSV format.
    #
    #  @param data The data, retrieved from the get() function.
    #
    #  @param fname The file name to store the data in.
    #
    #  @see get()
    #
    #######################################################################

    @staticmethod
    def toCSV(data,fname="output.csv"):
        
        with open(fname,'a') as file:
            file.write(",".join([str(j) for i in data for j in i]))


    #######################################################################
    #
    ## Function to write data to the database.
    #
    #  The write() function inserts new data into a table of the database.
    #
    #  @param table The name of the database's table to write to.
    #
    #  @param columns The columns to insert into, as a comma-separated string.
    #
    #  @param data The new data to insert, as a comma-separated string.
    #
    #######################################################################
                
    def write(self,table,columns,data):
        try:
            query = "INSERT OR REPLACE INTO {0} ({1}) VALUES ({2});".format(table,columns,data)

            self.cursor.execute(query)
        except Exception as e:
            logging.error('Database error:')
            logging.error(e)


    def executeWithRetries(self, query):
        maximumTries = 1000

        for i in range(0, maximumTries):        
            try:
                self.cursor.execute(query)
                
                # if it's here it means it succeeded
                break
            except sqlite3.OperationalError as e:
                if str(e) == 'database is locked':
                    logging.error(f'Database error. Retrying. {i + 1} of {maximumTries}.')
                    logging.error(e)
                
                    seconds = random.randrange(100, 1000) / 1000
                    time.sleep(seconds)
                else:
                    logging.error(f'Database error:')
                    logging.error(e)
                    break
        
        self.conn.commit()

    def insert(self, table, item):
        try:
            if not item:
                return

            columns = ''
            data = ''

            i = 0

            keys = item.keys()
            for key in keys:
                columns += key

                value = item[key]

                if isinstance(value, str):
                    value = "'" + value.replace("'", "''") + "'"
                elif value == None:
                    value = 'null'
                
                data += str(value)

                if i < len(keys) - 1:
                    columns += ', '
                    data += ', '                    

                i += 1

            query = "INSERT OR REPLACE INTO {0} ({1}) VALUES ({2});".format(table,columns,data)

            self.executeWithRetries(query)
        except Exception as e:
            logging.error(f'Database error:')
            logging.error(e)

    #######################################################################
    #
    ## Function to query any other SQL statement.
    #
    #  This function is there in case you want to execute any other sql
    #  statement other than a write or get.
    #
    #  @param sql A valid SQL statement in string format.
    #
    #######################################################################

    def execute(self, statement):
        self.executeWithRetries(statement)

    def query(self,sql):
        self.cursor.execute(sql)


    #######################################################################
    #
    ## Utility function that summarizes a dataset.
    #
    #  This function takes a dataset, retrieved via the get() function, and
    #  returns only the maximum, minimum and average for each column.
    #
    #  @param rows The retrieved data.
    #
    #######################################################################

    @staticmethod
    def summary(rows):
            
        # split the rows into columns
        cols = [ [r[c] for r in rows] for c in range(len(rows[0])) ]
        
        # the time in terms of fractions of hours of how long ago
        # the sample was assumes the sampling period is 10 minutes
        t = lambda col: "{:.1f}".format((len(rows) - col) / 6.0)

        # return a tuple, consisting of tuples of the maximum,
        # the minimum and the average for each column and their
        # respective time (how long ago, in fractions of hours)
        # average has no time, of course
        ret = []

        for c in cols:
            hi = max(c)
            hi_t = t(c.index(hi))

            lo = min(c)
            lo_t = t(c.index(lo))

            avg = sum(c)/len(rows)

            ret.append(((hi,hi_t),(lo,lo_t),avg))

        return ret
