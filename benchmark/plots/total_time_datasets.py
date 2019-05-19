# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  dataset_helper import DatasetHelper,color_for,better_name

def autolabel(ax,rects):
    """
    Attach a text label above each bar displaying its height
    """

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2. - .13, 1.0*height,
                '{:10.2f}'.format(round(height,2)),
                ha='center', va='bottom',size=10)
mpl.rc('hatch',color = "white")
with DatasetHelper() as dataset_helper:
    label_memory = ["Sem compressão","Com compressão"]
    for j,plan  in enumerate(["none","lzss"]):
        figs, axs = plt.subplots(1,1,False,False)
        ax = axs
        fig = figs
        modes = [x[0] for x in dataset_helper.select_data("SELECT distinct (mode || '_' ||threads) FROM result order by mode")]
        datasets = [x[0] for x in dataset_helper.select_data('''SELECT distinct dataset FROM result''')]
        ind = np.arange(len(datasets))  # the x locations for the groups

        width = 1#(1 - 0.1)/len(datasets)  # the width of the bars
        
        rects = []
        
        for i,mode in enumerate(modes):
            # print(mode,i)
            mean = []
            std = []
            for dataset in datasets:
                vec = dataset_helper.select_vector("SELECT total_time FROM result WHERE (mode || '_' ||threads) ='{0}' and dataset = '{1}' and compression='{2}' ".format(mode,dataset,plan))
                mean.append(vec.mean())
                std.append(vec.std())
            
            mean = [x / 1000.0 for x in mean]
            std = [x / 1000.0 for x in std]
            # print("Mean:",mean)
            # print("Std:",std)
            style = color_for(mode)
            color = style[0]
            rect = ax.bar(ind + (width * i), mean, width, yerr=std,error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),
                            label=mode,color=color
                            ,hatch=style[2]
                            )
            rects.append(rect)
            autolabel(ax,rect)
            
        # rects2 = ax.bar(ind + width/2, women_means, width, yerr=women_std,
        #                 color='IndianRed', label='Women')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Tempo(seg.)')
        ax.set_title(label_memory[j])


        ax.set_xticks( [i * width   for i,x in enumerate(modes)])
        ax.set_xticklabels([better_name(x) for x in  modes],rotation='45')
        # ax.set_yscale('log')
        # ax.legend()

        fig.set_size_inches(12.5, 5.5)
        fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)  # create some space below the plots by increasing the bottom-value

        # plt.yscale("log", nonposy="clip")
        plt.savefig("img/dedup_total_time_{0}.pdf".format(plan),bbox_inches='tight', transparent="True", pad_inches=0)
