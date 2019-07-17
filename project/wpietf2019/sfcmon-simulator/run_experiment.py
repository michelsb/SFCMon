#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import csv, os, sys
import pandas as pd
import numpy as np
import datetime
import rpy2.robjects as robjects
from countminsketch import CountMinSketch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding('utf8')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(ROOT_DIR, '..' )
BASE_DIR = os.path.join(ROOT_DIR, '..' )
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
CAIDA_DIR = os.path.join(DATASET_DIR, 'caida')

r = robjects.r
r.library("nortest")
r.library("MASS")

r('''
        wilcox.onesample.test <- function(v, verbose=FALSE) {
           wilcox.test(v,mu=median(v),conf.int=TRUE, conf.level = 0.95)
        }
        wilcox.twosamples.test <- function(v, r, verbose=FALSE) {
           wilcox.test(v,r)
        }
        tstudent.onesample.test <- function(v, verbose=FALSE) {
           t.test(v, mu = mean(v), alternative = "two.sided")
        }
        ''')

# Normality Test
lillie = robjects.r('lillie.test') # Lilliefors

# Close pdf graphics
close_pdf = robjects.r('dev.off')

# Non-parametric Tes
wilcoxon_test_two_samples = robjects.r['wilcox.twosamples.test']
wilcoxon_test_one_sample = robjects.r['wilcox.onesample.test']
t_test_one_sample = robjects.r['tstudent.onesample.test']

#metrics = ["fpr","recall","precision","accuracy","f1-score"]
metrics = ["fpr","recall","precision","accuracy"]
factors = ["time_window"]
metrics_position = {"fpr":10,"recall":8,"precision":12,"accuracy":13,"f1-score":14}
metrics_title = {"fpr":"False Positive Rate (%)","recall":"Recall (%)","precision":"Precision (%)","accuracy":"Accuracy (%)","f1-score":"F1-Score (%)"}
metrics_properties= {
    "fpr":{"title":"False Positive Rate","marker":">","color":"red"},
    "recall":{"title":"Recall","marker":"^","color":"green"},
    "precision":{"title":"Precision","marker":"s","color":"blue"},
    "accuracy":{"title":"Accuracy","marker":"x","color":"cyan"},
    "f1-score":{"title":"F1-Score","marker":"*","color":"magenta"}
}
# metrics_properties= {
#     "fpr":"m",
#     "recall":"c",
#     "precision":"r",
#     "accuracy":"g",
#     "f1-score":"b"}


# 0,1%
bound = 0.001
# window of 20 seconds (to clean up the sketch)
#array_num_chunks = [10,9,8,7,6,5,4,3,2,1]
array_num_chunks = [10,8,6,4,2,1]
num_rows_to_start = 1000
five_tuple_id = ["SrcIP", "DstIP", "Proto", "SrcPort", "DstPort"]
flow_id = ["SrcIP"]
colors = (0, 0, 0)

def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

def read_csv_file(file_name):
    print("# " + str(datetime.datetime.now()) + " - Reading CSV...")
    #df = pd.read_csv("../datasets/equinix-nyc.dirA.20180315-125910.UTC.anon.csv",sep=',',header=None)
    df = pd.read_csv(CAIDA_DIR+"/"+file_name, sep=',', header=None)
    df.columns = ["SrcIP", "DstIP", "Proto", "SrcPort", "DstPort","Size"]
    print("# " + str(datetime.datetime.now()) + " - CSV loaded inside a dataframe...")
    df["Size"] = df["Size"].apply(lambda x: x / 50)
    return df

def generate_baseline(df):
    print("#################### PROCESSING ALL DATASET #########################")
    baseline = {}
    num_packets = df.shape[0]
    print("# Number of packets = {}".format(num_packets))
    hh_threshold_packets = num_packets * bound
    total_size = df["Size"].sum()
    hh_threshold_size = total_size * bound
    print("# HH Threshold Packets = {}".format(hh_threshold_packets))
    print("# HH Threshold Size = {}".format(hh_threshold_size))

    df_flows = df.groupby(flow_id)
    df_flows_counts = df_flows.size().to_frame(name='counts')
    df_flows_stats = df_flows_counts \
            .join(df_flows.agg({"Size": "mean"}).rename(columns={'Size': 'size_mean'})) \
            .join(df_flows.agg({"Size": "median"}).rename(columns={'Size': 'size_median'})) \
            .join(df_flows.agg({"Size": "min"}).rename(columns={'Size': 'size_min'})) \
            .join(df_flows.agg({"Size": "sum"}).rename(columns={'Size': 'size_sum'})) \
            .reset_index()
    # print df_flows_stats.sort_values(by='size_sum', ascending=False).reset_index()
    shape = df_flows_stats.shape
    is_hh_packets = df_flows_stats["counts"] >= hh_threshold_packets
    is_hh_size = df_flows_stats["size_sum"] >= hh_threshold_size
    df_hh_flows_packets = df_flows_stats[is_hh_packets]
    df_hh_flows_size = df_flows_stats[is_hh_size]
    df_not_hh_flows_packets = df_flows_stats[~is_hh_packets]
    df_not_hh_flows_size = df_flows_stats[~is_hh_size]
    print("# Flow ID = {}".format(" ".join(str(x) for x in flow_id)))
    print("     Number of flows = {}".format(shape[0]))
    print("     Number of HH flows by num packets = {}".format(df_hh_flows_packets.shape[0]))
    print("     Number of HH flows by size = {}".format(df_hh_flows_size.shape[0]))
    baseline["packets"] = set(df_hh_flows_packets["SrcIP"].tolist())
    baseline["size"] = set(df_hh_flows_size["SrcIP"].tolist())
    baseline["tn_packets"] = set(df_not_hh_flows_packets["SrcIP"].tolist())
    baseline["tn_size"] = set(df_not_hh_flows_size["SrcIP"].tolist())
    return baseline

def simulate_sfcmon(df, num_chunks):
    print("#################### PROCESSING CHUNKS #########################")
    num_packets = df.shape[0]
    print("# Number of packets = {}".format(num_packets))
    chunks = split(df, num_packets / num_chunks)
    results = {"packets": set(), "size": set()}
    for c in chunks:
        print("# Chunk Data: {}; {}".format(c.shape, c.index))
        sketch_packets = CountMinSketch(5436, 5)  # table size=1000, hash functions=10
        sketch_size = CountMinSketch(5436, 5)  # table size=1000, hash functions=10
        count_packets = 0
        count_size = 0
        for row in zip(c["SrcIP"],c["Size"]):
            flow_id = row[0]
            sketch_packets.add(flow_id)
            sketch_size.add(flow_id, value=row[1])
            count_packets += 1
            count_size += row[1]
            if count_packets > num_rows_to_start:
                hh_threshold_packets = count_packets * bound
                hh_threshold_size = count_size * bound
                if sketch_packets[flow_id] > hh_threshold_packets:
                    results["packets"].add(flow_id)
                if sketch_size[flow_id] > hh_threshold_size:
                    results["size"].add(flow_id)
    return results

def calculate_metrics(TP, TN, FP, FN):

    ## Pre-calculations
    # PCP (Predicted condition positive) = TP+FP
    # PCN (Predicted condition negative) = FN+TN
    # CP (condition positive) = TP+FN
    # CN (condition negative) = TN+FP
    # TPOP (total population) =  CP + CN

    PCP = TP+FP
    PCN = FN+TN
    CP = TP+FN
    CN = TN+FP
    TPOP = CP+CN

    ## Metrics
    # True Positive Rate (TPR), also known as sensitivity or recall, is the proportion of flows who are HHs who are identified as HHs.
    TPR = recall = TP/(CP*1.0)
    # True Negative Rate (TNR), also known as specificity or selectivity, is the proportion of flows who are not HHs who are identified as not being HHs.
    TNR = TN/(CN*1.0)
    # False Positive Rate (FPR), also known as fall-out, is probability of false alarm, is the proportion of flows who are not HHs who are identified as HHs
    FPR = FP/(CN*1.0)
    # False Negative Rate (FPR), also known as miss rate, is the proportion of flows who are HHs who are identified as not being HHs
    FNR = FN/(CP*1.0)
    # Precision
    precision = TP/(PCP*1.0)
    # Accuracy
    accuracy = (TP+TN)/(TPOP*1.0)
    # F1 Score
    f1_score = 1/(((1/recall)+(1/precision))/2.0)

    print("     True Positive Rate (Sensitivity): " + str(100 * TPR))
    print("     True Negative Rate (Specifity): " + str(100 * TNR))
    print("     False Positive Rate: " + str(100*FPR))
    print("     False Negative Rate: " + str(100 * FNR))
    print("     Precision: " + str(100 * precision))
    print("     Accuracy: " + str(100 * accuracy))
    print("     F1-score: " + str(100 * f1_score))
    results = {}
    results["TPR"] = TPR
    results["TNR"] = TNR
    results["FPR"] = FPR
    results["FNR"] = FNR
    results["precision"] = precision
    results["accuracy"] = accuracy
    results["f1_score"] = f1_score

    return results

def flow_count_generation():
    out_file = csv.writer(open("flows_file.csv", "w"), delimiter=',', quoting=csv.QUOTE_ALL)
    columns = ["file_name", "total_packets", "num_flows", "num_hh_flows_packets", "num_hh_flows_size","num_five_tuples"]
    out_file.writerow(columns)

    file = open("csv_files.txt", "r")
    for line in file:
        print ("#### Processing file: " + line)
        df = read_csv_file(line.replace("\r", "").replace("\n", ""))
        num_packets = df.shape[0]
        baseline = generate_baseline(df)
        num_flows = len(baseline["packets"]) + len(baseline["tn_packets"])
        num_hh_flows_packets = len(baseline["packets"])
        num_hh_flows_size = len(baseline["size"])
        df_flows = df.groupby(five_tuple_id)
        df_flows_counts = df_flows.size().to_frame(name='counts')
        df_flows_stats = df_flows_counts \
            .join(df_flows.agg({"Size": "mean"}).rename(columns={'Size': 'size_mean'})) \
            .join(df_flows.agg({"Size": "median"}).rename(columns={'Size': 'size_median'})) \
            .join(df_flows.agg({"Size": "min"}).rename(columns={'Size': 'size_min'})) \
            .join(df_flows.agg({"Size": "sum"}).rename(columns={'Size': 'size_sum'})) \
            .reset_index()
        num_five_tuples = df_flows_stats.shape[0]
        data = [line,num_packets,num_flows,num_hh_flows_packets,num_hh_flows_size,num_five_tuples]
        out_file.writerow(data)

    file.close()


def perform_experiments():
    out_file_packets = csv.writer(open("stats_file_packets.csv", "w"), delimiter=',', quoting=csv.QUOTE_ALL)
    out_file_size = csv.writer(open("stats_file_size.csv", "w"), delimiter=',', quoting=csv.QUOTE_ALL)
    columns = ["num_chunks","file_name","total_num_flows","total_num_hh_flows","TP", "TN", "FP", "FN","TPR (Recall)","TNR","FPR","FNR","Precision","Accuracy","F1_Score"]
    out_file_packets.writerow(columns)
    out_file_size.writerow(columns)

    file = open("csv_files.txt", "r")
    for line in file:
        line = line.replace("\r", "").replace("\n", "")
        print ("#### Processing file: " + line)
        df = read_csv_file(line)
        baseline = generate_baseline(df)
        for num_chunks in array_num_chunks:
            print ("### Processing num chunks: " + str(num_chunks))
            results = simulate_sfcmon(df, num_chunks)

            print ("## Differences concerning number of packets")
            TP = len(results["packets"] & baseline["packets"]) # The flow was reported as HH and it is HH
            TN = len(baseline["tn_packets"] - results["packets"]) # The flow was not reported as HH and it is not HH
            FP = len(results["packets"] - baseline["packets"]) # The flow was reported as HH, but it is not HH.
            FN = len(baseline["packets"] - results["packets"]) # The flow was not reported as HH, but it is HH
            metrics = calculate_metrics(TP, TN, FP, FN)
            data_packets = [num_chunks,line,len(baseline["packets"])+len(baseline["tn_packets"]),len(baseline["packets"]),TP,TN,FP,FN,metrics["TPR"],metrics["TNR"],metrics["FPR"],metrics["FNR"],
                            metrics["precision"],metrics["accuracy"],metrics["f1_score"]]
            out_file_packets.writerow(data_packets)

            print ("## Differences concerning size")
            TP = len(results["size"] & baseline["size"])  # The flow was reported as HH and it is HH
            TN = len(baseline["tn_size"] - results["size"])  # The flow was not reported as HH and it is not HH
            FP = len(results["size"] - baseline["size"])  # The flow was reported as HH, but it is not HH.
            FN = len(baseline["size"] - results["size"])  # The flow was not reported as HH, but it is HH
            metrics = calculate_metrics(TP, TN, FP, FN)
            data_size = [num_chunks,line,len(baseline["size"])+len(baseline["tn_size"]), len(baseline["size"]), TP, TN, FP, FN, metrics["TPR"], metrics["TNR"], metrics["FPR"], metrics["FNR"],
                            metrics["precision"], metrics["accuracy"], metrics["f1_score"]]
            out_file_size.writerow(data_size)

    file.close()

def generate_stats():
    os.system("rm -rf ./figures ")
    os.system("mkdir -p ./figures")

    vectors_metrics = {}
    r_vectors_metrics = {}

    f = open("stats_file_packets.csv", 'rb')  # opens the csv file

    try:
        array_rates = []
        reader = csv.reader(f, delimiter=',')  # creates the reader object
        tester = True
        for row in reader:  # iterates the rows of the file in orders
            if tester:
                tester = False
                continue
            if ''.join(row).strip():
                num_chunks = row[0]
                if row[0] not in vectors_metrics:
                    vectors_metrics[num_chunks] = {}
                    r_vectors_metrics[num_chunks] = {}
                for metric in metrics:
                    if metric not in vectors_metrics[num_chunks]:
                        vectors_metrics[num_chunks][metric] = []
                        r_vectors_metrics[num_chunks][metric] = {}
                    vectors_metrics[num_chunks][metric].append(float(row[metrics_position[metric]]))

    finally:
        f.close()

    vector_medians = {}
    vector_errors = {}

    for metric in metrics:
        if metric not in vector_medians:
            vector_medians[metric] = []
            vector_errors[metric] = []
        for num_chunks in array_num_chunks:
            num_chunks = str(num_chunks)
            if metric == "recall":
                vector_medians[metric].append(100)
                vector_errors[metric].append(0)
            else:
                r_sample = robjects.FloatVector(vectors_metrics[num_chunks][metric])
                r_vectors_metrics[num_chunks][metric]["sample"] = r_sample
                t_test = t_test_one_sample(r_sample)
                error_max = t_test[3][1]
                mean = t_test[5][0]
                r_vectors_metrics[num_chunks][metric]["t_test_mean"] = mean
                r_vectors_metrics[num_chunks][metric]["t_test_error"] = float(error_max) - float(mean)
                vector_medians[metric].append(r_vectors_metrics[num_chunks][metric]["t_test_mean"]*100)
                vector_errors[metric].append(r_vectors_metrics[num_chunks][metric]["t_test_error"]*100)


    fig_num = 0
    x = np.arange(len(array_num_chunks))+1
    ylabel = "Performance Measures (%)"
    plt.figure(fig_num)
    fig, ax = plt.subplots()

    #time_window = [10, 9, 7, 5, 3, 1]

    for metric in metrics:
        ax.errorbar(x, vector_medians[metric],
                    yerr=vector_errors[metric], color=metrics_properties[metric]["color"],
                    marker=metrics_properties[metric]["marker"], mfc=metrics_properties[metric]["color"],
                    mec=metrics_properties[metric]["color"], ms=8, label=metrics_properties[metric]["title"])

    fig.text(0.5, 0.01, "Stream Time Window (seconds)", ha='center', fontsize=16)
    fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=16)

    #ax.set_xticklabels([60/i for i in array_num_chunks], fontsize=14)
    #print x
    ax.set_xticklabels([60/i for i in array_num_chunks], fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_xticks(x)
    ax.legend(prop={'size': 12})
    plt.savefig("./figures/metrics.pdf", format='pdf')

    plt.close()


    #flow_count_generation()
#perform_experiments()
generate_stats()



