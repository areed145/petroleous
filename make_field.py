import pandas as pd
import random

r = random.Random()
r.seed(42)

rows = []
for i in range(50000):
    loc = str(i)
    city = r.randint(0,5)
    doe = r.randint(0,100)
    crm = r.randint(0,100)
    obe = r.randint(0,100)
    att = r.randint(0,100)
    ned = r.randint(0,100)
    lat = r.randint(31000,33000)/1000
    lon = r.randint(-103000,-100000)/1000

    rows.append([loc,city,doe,crm,obe,att,ned,lat,lon])
    
df = pd.DataFrame(rows, columns=['Location Name',
							'City',
							'DOE Need (with Truancy)',
							'Crime',
							'Obesity',
							'6-8 attendance',
							'Need Score',
							'Latitude',
							'Longitude'])

df.to_csv('gen_data.csv')
