#!/usr/bin/env python3

from os.path import basename,splitext
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from dataset_helper import DatasetHelper

with DatasetHelper() as db:
    # {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
    patterns = ('//', '+', 'x', '.', '\\', 'o', '-', '*')
    patterns = patterns*5 #to repeat patterns
    light_colors = True
    legend = False

    for dataset in ["large","linux","silesia"]:
        sql_select = "select total_time from result where dataset like '%{0}%' and memory=0 and mode = '{1}' and threads={2}"
        data = OrderedDict()
        
        data['CPU-only'] =  OrderedDict()
        data['CPU-only']['Sequential'] = db.select_vector(sql_select.format(dataset,"sequential",0))
        data['CPU-only']['SPar'] = db.select_vector(sql_select.format(dataset,"spar",1))

        data['1 GPU'] =  OrderedDict()
        data['1 GPU']['CUDA'] =  db.select_vector(sql_select.format(dataset,"cuda",0))
        data['1 GPU']['OpenCL'] =  db.select_vector(sql_select.format(dataset,"opencl",0))
        
        data['1 GPU']['SPar+CUDA'] =  db.select_vector(sql_select.format(dataset,"cuda_spar_original",0))
        data['1 GPU']['SPar+OpenCL'] =  db.select_vector(sql_select.format(dataset,"opencl_spar_original",0))
        data['1 GPU']['SPar+CUDA (with batches)'] =  db.select_vector(sql_select.format(dataset,"cuda_spar",0))
        data['1 GPU']['SPar+OpenCL (with batches)'] =  db.select_vector(sql_select.format(dataset,"opencl_spar",0))
        data['1 GPU']['SPar+CUDA (2x mem. spaces)'] =  db.select_vector(sql_select.format(dataset,"cuda_spar_2xmemory",0))
        data['1 GPU']['SPar+OpenCL (2x mem. spaces)'] =  db.select_vector(sql_select.format(dataset,"opencl_spar_2xmemory",0))
       
        data['2 GPUs'] = OrderedDict()
        data['2 GPUs']["CUDA"] = db.select_vector(sql_select.format(dataset,"cuda",1))
        data['2 GPUs']["OpenCL"] = db.select_vector(sql_select.format(dataset,"opencl",1))

        
        data['2 GPUs']['SPar+CUDA'] =  db.select_vector(sql_select.format(dataset,"cuda_spar_original",1))
        data['2 GPUs']['SPar+OpenCL'] =  db.select_vector(sql_select.format(dataset,"opencl_spar_original",1))
        data['2 GPUs']['SPar+CUDA (with batches)'] =  db.select_vector(sql_select.format(dataset,"cuda_spar",1))
        data['2 GPUs']['SPar+OpenCL (with batches)'] =  db.select_vector(sql_select.format(dataset,"opencl_spar",1))
        data['2 GPUs']['SPar+CUDA (2x mem. spaces)'] =  db.select_vector(sql_select.format(dataset,"cuda_spar_2xmemory",1))
        data['2 GPUs']['SPar+OpenCL (2x mem. spaces)'] =  db.select_vector(sql_select.format(dataset,"opencl_spar_2xmemory",1))


        environments = list(data.keys())
        # environments = list(['CPU-only', '1 GPU','2 GPUs'])
        # versions = set(data[next(iter(data))].keys())
        versions = list()
        for env in data:
            for ver in data[env]:
                versions.append(ver)
        # versions = ['CUDA','SPar + CUDA',  'Sequential', 'SPar']
        versions = list(OrderedDict.fromkeys(versions).keys())

        print(environments)
        print(versions)
        # versions = list(versions.keys())

        seq_time = np.mean(data["CPU-only"]["Sequential"])

        cmap = plt.get_cmap('tab20')
        color_range = np.arange(10)*2
        if light_colors:
            color_range += 1
        colors = cmap(list(color_range)*5)

        ind = np.arange(len(environments))
        width = 0.3

        fig1, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.set_ylabel('Execution time (log(s))')
        ax2.set_ylabel('Speedup (x)')

        patches = [None] * len(versions)

        speedup = {
            'val': list(),
            'pos': list(),
        }
        pos = 0
        i = 0
        w_i = 0
        for env in environments:
            v_i = 0
            for ver in versions:
                if env in data and ver in data[env]:
                    value = np.mean(data[env][ver])
                    barcontainer = ax1.bar(
                        pos,
                        value,
                        label=ver,
                        yerr=np.std(data[env][ver]),
                        width=width,
                        bottom=0,
                        error_kw={'elinewidth': 1, 'capsize': 3},
                        color=colors[v_i],
                        edgecolor='w',
                        hatch=patterns[v_i] if patterns else None,
                    )
                    patches[v_i] = barcontainer.patches[0]

                    speedup['val'].append(seq_time/value)
                    speedup['pos'].append(pos)

                    # x,y = patches[v_i].get_xy()
                    # x += patches[v_i].get_width()/2
                    # y += patches[v_i].get_height()
                    # plt.annotate(ver, (x,y), va='bottom', ha='left', rotation=45)

                    i += 1
                    pos += width

                v_i += 1
            
            pos += width
            w_i += 1
            i += 1


        ax2.plot(speedup['pos'], speedup['val'], 'r.:')

        for pos,val in zip(speedup['pos'], speedup['val']):
            ax2.annotate(
                "%.1f" % val if val < 10 else "%.0f" % val,
                (pos, val),
                va='bottom',
                ha='center',
                xytext=(0, 3),
                textcoords='offset pixels',
                fontsize='small',
            )

        # ax1.set_xticks(np.array([width+1, (len(versions)+2)*width]))
        # ax1.set_xticklabels(list(map(lambda x: 'Workload ' + x, environments)))
        # ax1.set_xticklabels(environments)

        # ax1.set_xticks(np.array([width, (len(versions)+2)*width]))
        pos = 0
        ticks = list()
        for env in data:
            thispos = (len(data[env])-1)/2*width
            ticks.append(pos+thispos)
            pos += (len(data[env])+1)*width
        ax1.set_xticks(ticks)
        # ax1.set_xticks([1])
        ax1.set_xticklabels(environments)

        # ax1.set_xticks([width, (len(versions)+2)*width])
        # ax1.set_xticklabels((environments[0], environments[1]))

        # ax2.set_xticks(np.array([(len(versions)+3)*width]))
        # ax2.set_xticklabels((environments[1], ''))

        ax1.set_yscale('log')
        # ax2.set_yscale('log')

        ax1.legend(patches,
            versions,
            title=None,
            # loc="upper center",
            loc=(-0.1,-0.30),
            ncol=3,
            fancybox=False,
            handlelength=3,
            handleheight=1.2,
            # bbox_to_anchor=(0.5, 0, 0.5, 1)
        )

        # for i in range(0,10):
        #     ax1.bar(
        #         i,
        #         i*2,
        #         width=0.7,
        #         bottom=0,
        #         # color=colors[i],
        #         # edgecolor='w',
        #         # hatch=patterns[i] if patterns else None,
        #     )

        fig1.set_size_inches(8, 6)

        filename = splitext(basename(__file__))[0]

        plt.title(dataset.capitalize())
        for fmt in ('png','pdf'): #png, pdf, ps, eps, svg
            plt.savefig(fmt + '/' + filename + "_"+dataset +'.' + fmt, dpi=100, format=fmt, bbox_inches='tight')
        # plt.show()
        plt.close()
