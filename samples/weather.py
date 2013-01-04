from blaze import open
from blaze import mean, std

from datetime import datetime

table = open('ctable://noaa_gsod_example')
print table

#------------------------------------------------------------------------

start = datetime(year = 2008 , month = 1 , day = 1)
end   = datetime(year = 2009 , month = 1 , day = 1)

res = table.select(lambda x: start < x < end, col='YEARMODA')

mean(res, 'TEMP')
std(res, 'TEMP')

#------------------------------------------------------------------------

start = datetime(year = 2009 , month = 1 , day = 1)
end   = datetime(year = 2010 , month = 1 , day = 1)

res2 = table.select(lambda x: start < x < end, col='YEARMODA')

mean(res2, 'TEMP')
std(res2, 'TEMP')
