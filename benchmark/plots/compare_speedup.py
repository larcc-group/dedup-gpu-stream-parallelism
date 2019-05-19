
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  dataset_helper import DatasetHelper,available_colors
import os

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
# Colorblind-friendly colors
colors = [[0,0,0], [230/255,159/255,0], [86/255,180/255,233/255], [0,158/255,115/255], 
          [213/255,94/255,0], [0,114/255,178/255]]

sql = """
select dataset,mode,threads as threads, (select avg(total_time) from result as result_sequential where   result_sequential.mode = 'sequential') as sequential_time, avg(total_time)  as avg_time, STDEV(total_time) as stdev

 from result where  mode <> 'sequential'
AND mode <> 'spar'  -- AND dataset like "{0}"
 group by dataset,mode,threads
 
 UNION 
SELECT dataset,mode,threads,sequential_time, avg_time,stdev FROM ( 
select dataset,mode,threads  as threads, (select avg(total_time) from result as result_sequential where   result_sequential.mode = 'sequential' ) as sequential_time, avg(total_time)  as avg_time, STDEV(total_time) as stdev

 from result where  mode = 'spar'  -- AND dataset like "{0}"
 group by dataset,mode,threads
 order by 5  limit 1
) AS T 

 order by dataset,mode,threads
"""

datasets = [
    {"name":"Simlarge", "dataset_name":"/home/charles/parsec/parsec-3.0/pkgs/kernels/dedup/inputs/input_simlarge.tar", "file_name":"simlarge"}
]

def autolabel(ax,rects):
    """
    Attach a text label above each bar displaying its height
    """

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2. - .13, 1.0*height,
                '{:10.2f}'.format(round(height,2)),
                ha='center', va='bottom',size=10)

def format_name(name, threads):
    if name == "spar":
        return "SPar("+str(threads)+" THREAD"+("S)" if str(threads) != "1" else ")")
    return name.replace("GPU_","")+"("+str(threads)+" GPU"+("S)" if threads > 1 else ")")
    
with DatasetHelper() as db:
    for dataset in datasets:
        data = [x for x in db.select_data(sql.format(dataset["dataset_name"]))]
        labels = [format_name(x[1] ,x[2] ) for x in data]
        print(labels)
        y_pos = np.arange(len(labels))  
        print(y_pos)   
        values = [x[3]/x[4] for x in data]
        y_error = [ ( x[3] / (x[4] + x[5] ) - x[3]/x[4]) if x[5] != None else 0  for x in data]
        print(y_error)
        plt.rcdefaults()
        fig, ax = plt.subplots()
        rect = ax.bar(y_pos, values,align='center',yerr = y_error ,color=colors)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title("Speedup")
        autolabel(ax,rect)
        
        plt.savefig("img/compare_speedup.png".format(dataset["file_name"]),bbox_inches='tight', pad_inches=0)
        # plt.show()