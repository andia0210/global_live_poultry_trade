#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


cd /


# In[3]:


cd /Users/andia0210/Desktop/Map


# In[4]:


trademap = pd.read_csv('US-EU_to_46-key-countries_1962-2019.csv', sep=',')


# In[5]:


trademap


# In[6]:


trademap_reporter=trademap.groupby(['Reporter_Countries', 'Startlon', 'Startlat', 'Reporter_Continent', 'Reporter_Region','Partner_Countries', 'Arrlon', 'Arrlat', 'Partner_Continent', 'Unit'])['Export_Value_corrected'].sum().reset_index()


# In[7]:


trademap_reporter


# In[8]:


trademap_reporter_sort=trademap_reporter.sort_values(by=['Reporter_Countries'])


# In[9]:


trademap_reporter_sort


# In[10]:


trademap_reporter_sort.to_csv (r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_reporter_sort.csv', index = False, header=True)


# In[11]:


trademap_partner=trademap.groupby(['Reporter_Region','Partner_Countries', 'Arrlon', 'Arrlat', 'Partner_Continent', 'Unit'])['Export_Value_corrected'].sum().reset_index()


# In[12]:


trademap_partner


# In[13]:


trademap_partner_sort=trademap_partner.sort_values(by=['Partner_Countries'])


# In[14]:


trademap_partner_sort


# In[15]:


trademap_partner_sort.to_csv (r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_partner_sort.csv', index = False, header=True)


# In[16]:


trademap_partner_sort['Ratios'] = 100 * trademap_partner_sort['Export_Value_corrected'] / trademap_partner_sort.groupby(['Partner_Countries', 'Arrlon', 'Arrlat', 'Partner_Continent', 'Unit'])['Export_Value_corrected'].transform('sum')

# Transformation: While aggregation must return a reduced version of the data, transformation can return some transformed version of the full data to recombine. For such a transformation, the output is the same shape as the input.


# In[17]:


trademap_partner_sort['Total_Export'] = trademap_partner_sort.groupby(['Partner_Countries', 'Arrlon', 'Arrlat', 'Partner_Continent', 'Unit'])['Export_Value_corrected'].transform('sum')

# Transformation: While aggregation must return a reduced version of the data, transformation can return some transformed version of the full data to recombine. For such a transformation, the output is the same shape as the input.


# In[18]:


trademap_partner_sort


# In[19]:


trademap_partner_sort.to_csv (r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_partner_add_sort.csv', index = False, header=True)


# In[20]:


trademap_partner_piechart_sort=trademap_partner_sort.groupby(['Partner_Countries', 'Arrlon', 'Arrlat', 'Partner_Continent', 'Unit'])['Export_Value_corrected'].sum().reset_index()


# In[21]:


trademap_partner_piechart_sort


# In[22]:


trademap_partner_piechart_sort.to_csv (r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_partner_piechart_totalexport.csv', index = False, header=True)


# In[23]:


trademap


# In[24]:


trademap_export_total=trademap.groupby(['Reporter_Countries', 'Startlon', 'Startlat', 'Reporter_Continent', 'Reporter_Region', 'Unit'])['Export_Value_corrected'].sum().reset_index()


# In[25]:


trademap_export_total


# In[26]:


trademap_export_total.to_csv (r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_reporter_totalexport.csv', index = False, header=True)


# In[27]:


pwd


# In[28]:


cd ../../Anaconda3/envs/Python/


# In[29]:


# conda install basemap
# conda install folium
# conda install the GDAL library
### conda install -c conda-forge gdal


# In[30]:


matplotlib


# In[31]:


# pip install geonamescache


# In[32]:


# Load Libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm

from geonamescache import GeonamesCache
# from helpers import slug
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from mpl_toolkits.basemap import Basemap


# Load the data (on the Windows)
data_reporter = pd.read_csv(
    r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_reporter_sort.csv', sep=",")

data_partner = pd.read_csv(
    r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_partner_add_sort.csv', sep=",")

piechart_data = pd.read_csv(
    r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_partner_piechart_totalexport_20kUP.csv', sep=",")

data_reporter_export = pd.read_csv(
    r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_reporter_totalexport_addcode.csv', sep=",")

country = pd.read_csv(
    r'C:\Users\andia0210\Desktop\Map\trade_country_coordinates_mdf.csv', sep=",")

shapefile = '/Users/andia0210/Desktop/Map/ne_10m_admin_0_countries/ne_10m_admin_0_countries'


num_colors = 4

filename = r'C:\Users\andia0210\Desktop\Map\trade_map_USEU_46Key_reporter_totalexport_addcode.csv'

cols = ['Country_Code', 'Reporter_Countries', 'Export_Value_corrected']
#title = 'Live Chicken Intercontinental Export, 1962-2019'
#imgfile = 'img/{}.png'.format(slug(title))

# description = '''Live chicken trade map'''.strip()


# In[33]:


gc = GeonamesCache()
iso3_codes = list(gc.get_dataset_by_key(gc.get_countries(), 'iso3').keys())


# In[34]:


df_1 = pd.read_csv(filename, usecols=cols)


# In[35]:


df_1


# In[36]:


df_1.set_index('Country_Code', inplace=True)


# In[37]:


df_1


# In[38]:


country


# In[39]:


num_colors = 4
cm = plt.get_cmap('Reds')
scheme = [cm(i / num_colors) for i in range(num_colors)]
colorscheme = [0,2,3,4]


# In[40]:


rc = df_1['Reporter_Countries'] 
values = df_1['Export_Value_corrected']


# In[41]:


scheme


# In[42]:


len(rc)


# In[43]:


bins = np.power(100, colorscheme)


# In[44]:


bins


# In[45]:


df_1['Bin'] = np.digitize(values, bins) - 1
df_1.sort_values('Bin', ascending=False)
# df_1.sort_values('bin', ascending=False).head(10)


# In[46]:


# Create the figure and set the dimension
my_dpi=150
fig, ax = plt.subplots(figsize=(1200/my_dpi, 900/my_dpi), dpi=my_dpi) 
#fig.suptitle('Live Chicken Intercontinental Export, 1962-2019', fontsize=20, weight='bold', family='times new roman', y=.95)

#fig = plt.figure(figsize=(22, 12))
#ax = fig.add_subplot(111, axisbg='w', frame_on=False)
#plt.title('Global Trade Map', size=20, weight='bold', family='times new roman') 


# Make the background World map
##map = Basemap(llcrnrlon=-0.5,llcrnrlat=39.8,urcrnrlon=4.,urcrnrlat=43., resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
##map = Basemap(resolution='c', llcrnrlon=-160, llcrnrlat=-80, urcrnrlon=160, urcrnrlat=80)
##map = Basemap(projection='robin', lon_0=0, resolution='c')
##map = Basemap(lon_0=0, projection='robin', resolution='c', llcrnrlon=-160, llcrnrlat=-80, urcrnrlon=160, urcrnrlat=80) 
map = Basemap(resolution='c', llcrnrlon=-170, llcrnrlat=-90, urcrnrlon=190, urcrnrlat=90)

##map.drawmapboundary(fill_color='#f2f2f2', linewidth=0)
map.drawmapboundary(color='w')

##map.fillcontinents(color='white',lake_color='#A6CAE0', alpha=0.3)
map.fillcontinents(color='grey',lake_color='#A6CAE0', alpha=0.3)
#map.fillcontinents(color='#DCDCDC',lake_color='#A6CAE0', alpha=0.3)


##map.drawcoastlines(linewidth=0.1, color='black')
map.drawcoastlines(linewidth=0.1, color='white')

##map.drawcountries(linewidth=0.1, color='grey')
map.drawcountries(linewidth=0.3, color='white')

#color code: USA->light blue: #87CEEB    EU->light green: #90EE90
#color code: USA->blue: #0097C6 or #002A93   EU->green: #00EA00 


# In[47]:


# Cover up Antarctica so legend can be placed over it.
plt.axhspan(-60, -90, facecolor='w', edgecolor='w',zorder=1)


# In[48]:


# Custom adjust of the subplots
# Adjust the scaling factor to fit your legend text completely outside the plot
# (smaller value results in more space being made for the legend)
plt.subplots_adjust(left=0.10,right=0.90,top=0.90,bottom=0.15,wspace=0.15,hspace=0.05)


# In[49]:



##map.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
map.readshapefile(shapefile, 'units', color='white', linewidth=.2, zorder=2)

for info, shape in zip(map.units_info, map.units):
    iso3 = info['ADM0_A3']
    if iso3 not in df_1.index:
        #color = '#dddddd'
        color = '#1C00ff00'
    else:
        color = scheme[df_1.loc[iso3]['Bin']]

    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches)
    pc.set_facecolor(color)
    ax.add_collection(pc)


# In[50]:


#########################################################################
# Add a marker per Reporter_country of the data frame                   #
# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html  #
#########################################################################

plt.plot(data_reporter['Startlon'], data_reporter['Startlat'], linestyle='none', marker="^", markersize=2, 
         alpha=0.6, c="purple", markeredgecolor="purple", markeredgewidth=0.4, zorder=3)


# In[51]:


# Add an arrow from Reporter_country to Partner_country one by one on the map
for i in range(0, len(data_reporter)):
    x1=data_reporter.iloc[i]['Startlon']
    y1=data_reporter.iloc[i]['Startlat']
    x2=data_reporter.iloc[i]['Arrlon']
    y2=data_reporter.iloc[i]['Arrlat']
    dx=data_reporter.iloc[i]['Arrlon']-data_reporter.iloc[i]['Startlon']
    dy=data_reporter.iloc[i]['Arrlat']-data_reporter.iloc[i]['Startlat']
    
    #reporter_loc=[x1, y1],
    #partner_loc=[x2, y2],
    
    if x1==-95.712891 and y1==37.09024:
        #plt.arrow(x1, y1, dx, dy, length_includes_head=True, linewidth=0.6, head_width=1, head_length=1.5, color='#87CEEB', alpha=0.3)
        #plt.arrow(x1, y1, dx, dy, length_includes_head=True, linewidth=0.6, head_width=1, head_length=1.5, color='#002A93', alpha=0.16, zorder=4)
        plt.arrow(x1, y1, dx, dy, length_includes_head=True, linewidth=0.8, head_width=1, head_length=1.5, color='navy', alpha=0.15, zorder=4)
    else:
        #plt.arrow(x1, y1, dx, dy, length_includes_head=True, linewidth=0.6, head_width=1, head_length=1.5, color='#90EE90', alpha=0.3)
        plt.arrow(x1, y1, dx, dy, length_includes_head=True, linewidth=0.6, head_width=1, head_length=1.5, color='#00EA00', alpha=0.16, zorder=4)
        
#color code: USA->light blue: #87CEEB    EU->light green: #90EE90
#color code: USA->blue: #0097C6 or #002A93   EU->green: #00EA00 


# In[52]:



# Draw color legend.
ax_legend = fig.add_axes([0.3, 0.2, 0.4, 0.03], zorder=3)
cmap = mpl.colors.ListedColormap(scheme)
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=colorscheme, boundaries=colorscheme, orientation='horizontal')
cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])

# Set the map footer.
#plt.annotate(description, xy=(0.3, 0.2), size=14, xycoords='axes fraction')


# In[53]:


# Prepare a color for each part of the pie chart depending on the Reporter_Region (EU, USA).

N=len(piechart_data['Export_Value_corrected'])
global_data = np.zeros((N,2))   # N = the number of pie charts on the map; 3 means the number of colors within each pie chart
global_data[:,0] = 1
global_data[:,1] = 2

#colors = ['#00EA00', '#002A93']
colors = ['#00EA00', 'navy']
font_size = 8

# Define pie chart locations
xlon = piechart_data['Arrlon']
ylat = piechart_data['Arrlat']

# Define sizes of the scatter marker (pie chart)
sizes = piechart_data['Export_Value_corrected']

gr = np.zeros((N,1))
global_ratio = {'Partner_Countries': piechart_data['Partner_Countries'],
                'EU': gr[:,0],
                'USA': gr[:,0],
               }

df = pd.DataFrame(global_ratio, columns = ['Partner_Countries', 'EU', 'USA'])


#color code: USA->light blue: #87CEEB    EU->light green: #90EE90
#color code: USA->blue: #0097C6 or #002A93   EU->green: #00EA00 


# In[54]:


df


# In[55]:


N


# In[56]:



# Function to draw pie charts on the map
def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

def draw_pie(ax, X=0, Y=0, size = 1000):
    
    #for b in range(N): 
    pc = df['Partner_Countries'].iloc[b]
    data_subset = data_partner[data_partner['Partner_Countries'] == pc]
    phase = data_subset['Reporter_Region']
    for p in phase:
        if p == 'EU':
            position = getIndexes(data_subset, p)[0][0]
            r = data_partner['Ratios'].iloc[position]
            df['EU'].iloc[b] = r/100
                
        elif p == 'USA':
            position = getIndexes(data_subset, p)[0][0]
            r = data_partner['Ratios'].iloc[position]
            df['USA'].iloc[b] = r/100

        else:
            df['EU'].iloc[b] = 0.0
            df['USA'].iloc[b] = 0.0
                
    xy = []
    s = []
    start = 0.0 
    ratios = [df['EU'].iloc[b], df['USA'].iloc[b]]
    for ratio in ratios:
        x = [0] + np.cos(np.linspace(2*np.pi*start,2*np.pi*(start+ratio), 60)).tolist() 
        y = [0] + np.sin(np.linspace(2*np.pi*start,2*np.pi*(start+ratio), 60)).tolist() 
        
        xy1 = np.column_stack([x, y])
        s1 = np.abs(xy1).max()          
        
        xy.append(xy1)
        s.append(s1)
        start += ratio 

    piecolors = []
    for j in range(0, 2):
        c = global_data[b,j]-1
        c = int(c)
        piecolors.append(colors[c])
        
    for xyi, si, i in zip(xy, s, range(2)):
        #ax.scatter([X],[Y] , marker=(xyi,0), s=size*si**2, facecolor=piecolors[i], edgecolor='k', linewidth=0.5, alpha=.7)
        ax.scatter([X],[Y] , marker=(xyi,0), s=size*si**2, facecolor=piecolors[i], edgecolor='k', linewidth=0.5, alpha=0.4, zorder=5)

# Plot pie charts:
for b in range(N):
    X,Y=map(xlon[b],ylat[b]) 
    Size = sizes[b]
    draw_pie(ax, X, Y, size = Size/2400) 


# In[57]:


plt.show()


# In[58]:


# Create the legend

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


legend_elements1 = [Patch(facecolor='#00EA00', edgecolor='black', label='Europe export'),
                   Patch(facecolor='navy', edgecolor='black', label='USA export')]

legend_elements2 = [Line2D([0], [1], ls='-', marker='>', lw=1, color='#00EA00', label='Europe Export'),
                   Line2D([0], [1], ls='-', marker='>', lw=1, color='navy', label='USA Export')]


legend1 = ax.legend(handles=legend_elements1, 
                    loc='lower left',
                    prop={'size': 6},
                    borderaxespad=2,    # Small spacing around legend box 
                    title="Exporter"  # Title for the legend
                   )    

ax.add_artist(legend1)

legend2 = ax.legend(handles=legend_elements2, 
                    loc='lower right', 
                    prop={'size': 6},
                    borderaxespad=2,    # Small spacing around legend box 
                    title="Trade"  # Title for the legend
                   )    

#color code: USA->light blue: #87CEEB    EU->light green: #90EE90
#color code: USA->blue: #0097C6 or #002A93   EU->green: #00EA00 


# In[59]:


# Save figure as png at 'C:\Users\andia0210\Anaconda3\envs\Python\PNG'
plt.savefig('PNG/Basemap_trademap_piechart_withoutlabel_06-02_navy_greymap.png', dpi=600, bbox_inches='tight')


# In[60]:


# Save figure as PDF at 'C:\Users\andia0210\Anaconda3\envs\Python\PNG'
plt.savefig('PNG/Basemap_trademap_piechart_withoutlabel_06-02_navy_greymap.pdf', dpi=600, bbox_inches='tight')


# In[ ]:


#########################################################################
# Add country label one by one on the map                               #
# https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.text     #
#########################################################################

reporter_countries = []
partner_countries = []
countrylist = []

#reporter_countries = data_reporter['Reporter_Countries']
#partner_countries = data_partner['Partner_Countries']

#reporter_countries = ['United States of America','United Kingdom','Netherlands','France','Germany','Spain']
reporter_countries = ['United States of America','United Kingdom','Netherlands','France','Germany','Spain','Austria','Belgium','Denmark','Hungary']



countrylist.extend(reporter_countries)
#countrylist.extend(partner_countries)
countrylist = list(dict.fromkeys(countrylist))

for i in range(0, len(country)):
    for cl in countrylist:
        if cl == country['Country'].iloc[i]:
            x=country.iloc[i]['Longitude']
            y=country.iloc[i]['Latitude'] 
            name=country.iloc[i]['Country']
            plt.text(x, y, name, fontsize=4, horizontalalignment='right', verticalalignment='top')
        

