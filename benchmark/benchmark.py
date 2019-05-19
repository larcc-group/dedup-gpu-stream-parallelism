#!/usr/bin/python3
from __future__ import print_function
import os
import glob
import re
from time import gmtime, strftime,sleep
from subprocess import Popen, PIPE
import sqlite3
from sys import argv
from datetime import datetime

GPUS = [0,1]
# PLANS = ["spar"]
# PLANS = ["cuda","opencl","sequential","cuda_spar","opencl_spar","spar"]
# PLANS = ["sequential","cuda_spar","opencl_spar","spar"]
# PLANS = ["sequential"]
# PLANS = ["sequential","cuda_spar","opencl_spar","spar"]
PLANS = ["sequential","cuda_spar","opencl_spar","spar"]

APP_FILE = "dedup"
SPAR_THREADS = 20
dir_name = os.path.dirname(__file__)
EXE_PATH =os.path.abspath(os.path.join(dir_name, "../src"))
COMPRESSIONS = ["lzss"]

THREADS_CPU = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

gpus_arg =   ",".join([str(x) for x in GPUS])
print(gpus_arg)
def log(text):
    print(text)
    with open("benchmark.log","a+") as log_file:
        log_file.write("{0}: {1}\n".format(datetime.now(),text))
    

class ItemResult:
    def __init__(self):
        self.mode = ""
        self.dataset = ""
        self.total_time = 0.0
        self.compression = "none"
        self.memory = 0

def main(args):
    if len(args) < 3:
        print("Usage: {0} <dataset> <repetitions> ".format(args[0]))
        return

    log("Dataset {0}".format(args[1]))
    log("Repetitions {0}".format(int(args[2])))
    run_for_benchmark("benchmark.db",args[1],int(args[2]))

def run_for_benchmark( database_name,dataset, repetitions):
    db = sqlite3.connect(database_name)
    # Get a cursor object
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS result(
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            dataset TEXT,
            mode TEXT,
            compression TEXT,
            seq int,
            threads int,
            total_time decimal(10,6),
            memory int,
            current_date TEXT,
            current_time TEXT
            )
    ''')
    db.commit()
    
    total = repetitions * len(PLANS)
    current = 0
    for i in range(repetitions):
        for plan in PLANS:
            for compression in COMPRESSIONS:
                #try again 3 times
                for retrial in range(3):
                    try:
                        num_threads = 1
                        if plan == "spar":
                            num_threads = len(THREADS_CPU)
                        elif  plan != 'sequential':
                            num_threads = len(GPUS)
                        for thread in range( num_threads):
                            tt = thread
                            
                            gpus_arg = "0"
                            if plan =="cuda_spar" or plan == "opencl_spar":
                                
                                gpus_arg =   ",".join([str(GPUS[i]) for i in range(0,thread + 1)])

                            #if plan =="dedup_cuda_spar" or plan == "dedup_opencl_spar":
                            if plan == "spar":
                                tt = THREADS_CPU[thread]    
                            elif "cuda" in plan or "opencl" in plan:
                                tt = 1
                            else:
                                tt = tt + 1
                            
                            in_file_result = exec_command(["dedup_"+plan,"-c","-i",os.path.abspath(os.path.join(dir_name,dataset)),"-o",os.path.join(dir_name,"temp.lz"),"-w",compression, "-t", str(tt),"-g",gpus_arg])
                            result = process_output(in_file_result)
                            result.seq = i
                            result.dataset = dataset
                            result.compression = compression
                            result.mode = plan
                            result.memory = 0
                            result.threads = thread
                            insert_result(db,result)


                            in_memory_result = exec_command(["dedup_"+plan,"-c","-i",os.path.abspath(os.path.join(dir_name,dataset)),"-o",os.path.join(dir_name,"temp.lz"),"-w",compression, "-t", str(tt),"-g",gpus_arg,"-p"])
                            result = process_output(in_memory_result)
                            result.seq = i
                            result.dataset = dataset
                            result.compression = compression
                            result.memory = 1
                            result.mode = plan
                            result.threads = thread
                            insert_result(db,result)

                        
                        break
                    except Exception as err:
                        log("An error ocurred: {0}".format(err))
                        sleep(1)
                        log("Trying again for {0} time".format(retrial + 1))
                
            current = current +1
            log("PROGRESSO: {0:.2f}".format(current/total * 100))


    db.close()

def insert_result(db,item):
    cursor = db.cursor()
    cursor.execute("""INSERT INTO result(
        dataset ,
            mode,
            compression,
            seq ,
            total_time ,
            threads ,
            memory,
            current_date,
            current_time
         ) VALUES (?,?,?,?,?,?,?,date('now'),time('now')) """,(
                item.dataset,
                item.mode,
                item.compression,
                item.seq,
                item.total_time,
                item.threads,
                item.memory
            ))
    db.commit()
        

def exec_command(command):
    
    app_path = os.path.join(EXE_PATH,command[0])
    # print("CMD:" ," ".join(command))
    # return ""
    app_path = os.path.join(EXE_PATH,command[0])
    process = Popen([app_path] + command[1:], stdout=PIPE, cwd=EXE_PATH)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        raise Exception("Wrong result for command '%s', output is %s " % (" ".join(command),output))
    log("CMD: "+" ".join(command))
    return output.decode('ascii')

def process_output(text):
    result = ItemResult()
    print(text)
    result.total_time = float(find_item(r"It took ([\d\.]+)",text,0))
    return result

    
def find_item(pattern, text, default):
    found = re.findall(pattern,text)
    if len(found) == 0:
        return default
    return found[0]
if __name__ == "__main__":
    main(argv)
    # print(vars(result))
