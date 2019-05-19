import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'Helvetica',
       
        'size'   : 20}


hfont = {'fontname':'Helvetica','fontsize':20}

with open("rabin_fingerprint_size_large.txt","r") as file:
    lines = [int(x) for x in file.readlines()]


x = lines
x = pd.Series(x)
x.hist(bins=50,edgecolor='black', linewidth=1, hatch="//",rwidth=.8)
# histogram on linear scale
# plt.subplot(211)
# hist, bins, _ = plt.hist(x, bins=50)

# histogram on log scale. 
# Use non-equal bin sizes, such that they look equal on log scale.
# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
# plt.subplot(212)
# plt.hist(x, bins=logbins)
# plt.xscale('log')
plt.ylabel("Ocorrências",**hfont)
plt.xlabel("Tamanho do segmento(bytes)",**hfont)



ax = plt.gca()

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12) 

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12) 
    
    # specify integer or one of preset strings, e.g.
    #tick.label.set_fontsize('x-small') 
    # tick.label.set_rotation('vertical')
plt.xlim(0,15000)
plt.show()