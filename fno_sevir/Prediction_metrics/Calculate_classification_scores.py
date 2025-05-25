
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .Classification_scores import *
import pandas as pd

def calculate_classification_scores(dir, number_test_seq, seq_size, csv_filename):
    np_files = [f for f in os.listdir(dir) if f.endswith('.npy')]
    np_files.sort()

    pred_file = np_files[number_test_seq:]
    gt_file = np_files[:number_test_seq]

    # Define metric names
    metrics = [
        "false_alarm_rate", "critical_success_rate", "true_skill_statistics",
        "prob_of_detection", "spec", "positive_predictive_value",
        "negative_predictive_value", "AUC", "AUPRC", "bias"
    ]

    # Initialize dictionaries for storing metric values
    metric_f = {m: [] for m in metrics}
    metric_f1 = {f"{m}(1)": [] for m in metrics}
    metric_f2 = {f"{m}(2)": [] for m in metrics}

    for i in range(number_test_seq):
        # Finding pred and gt file address
        gt_add = os.path.join(dir, gt_file[i])
        pred_add = os.path.join(dir, pred_file[i])

        ground_truth = np.load(gt_add)
        predicted = np.load(pred_add)

        ground_truth = ground_truth[0]
        predicted = predicted[0]

        # Temporary lists for per-sequence scores
        metric_a = {m: [] for m in metrics}

        for index in range(ground_truth.shape[0]):
            gt = ground_truth[index, :, :]
            pred = predicted[index, :, :]

            gt_binary = convert_to_binary_g_mean(gt)
            pred_binary = convert_to_binary_g_mean(pred)

            cs = ClassificationScores(gt_binary, pred_binary)

            # Store individual metric values
            metric_a["false_alarm_rate"].append(cs.false_alarm_rate())
            metric_a["critical_success_rate"].append(cs.critical_success_rate())
            metric_a["true_skill_statistics"].append(cs.true_skill_statistics())
            metric_a["prob_of_detection"].append(cs.prob_of_detection())
            metric_a["spec"].append(cs.spec())
            metric_a["positive_predictive_value"].append(cs.positive_predictive_value())
            metric_a["negative_predictive_value"].append(cs.negative_predictive_value())
            metric_a["AUC"].append(cs.AUC())
            metric_a["AUPRC"].append(cs.AUPRC())
            metric_a["bias"].append(cs.bias())

        # Compute mean values for full, first half, and second half
        for m in metrics:
            metric_f[m].append(np.mean(metric_a[m]))
            metric_f1[f"{m}(1)"].append(np.mean(metric_a[m][:seq_size // 2]))  # First half
            metric_f2[f"{m}(2)"].append(np.mean(metric_a[m][seq_size // 2:]))  # Second half

        final_results = {}
        for m in metrics:
            final_results[m] = np.mean(metric_f[m])
            final_results[f"{m}(1)"] = np.mean(metric_f1[f"{m}(1)"])  # First half
            final_results[f"{m}(2)"] = np.mean(metric_f2[f"{m}(2)"]) 

    # Prepare row data for results.csv
    results = {**final_results}
    results["dir"] = dir.split("/")[-1]  # Extract parent directory name as an identifier

    # Convert row data to a DataFrame
    df_new = pd.DataFrame([results])


    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new  # Create new file if it doesn't exist

    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")



csv_filename = "results.csv"

# Define columns
columns = [
    "false_alarm_rate", "critical_success_rate", "true_skill_statistics",
    "prob_of_detection", "spec", "positive_predictive_value",
    "negative_predictive_value", "AUC", "AUPRC", "bias",

    "false_alarm_rate(1)", "critical_success_rate(1)", "true_skill_statistics(1)",
    "prob_of_detection(1)", "spec(1)", "positive_predictive_value(1)",
    "negative_predictive_value(1)", "AUC(1)", "AUPRC(1)", "bias(1)",

    "false_alarm_rate(2)", "critical_success_rate(2)", "true_skill_statistics(2)",
    "prob_of_detection(2)", "spec(2)", "positive_predictive_value(2)",
    "negative_predictive_value(2)", "AUC(2)", "AUPRC(2)", "bias(2)"
]

# Create an empty DataFrame with the required columns
if not os.path.exists(csv_filename):
    df = pd.DataFrame(columns=["dir"] + columns)
    df.to_csv(csv_filename, index=False)



# Define columns
columns = [
    "false_alarm_rate", "critical_success_rate", "true_skill_statistics",
    "prob_of_detection", "spec", "positive_predictive_value",
    "negative_predictive_value", "AUC", "AUPRC", "bias",

    "false_alarm_rate(1)", "critical_success_rate(1)", "true_skill_statistics(1)",
    "prob_of_detection(1)", "spec(1)", "positive_predictive_value(1)",
    "negative_predictive_value(1)", "AUC(1)", "AUPRC(1)", "bias(1)",

    "false_alarm_rate(2)", "critical_success_rate(2)", "true_skill_statistics(2)",
    "prob_of_detection(2)", "spec(2)", "positive_predictive_value(2)",
    "negative_predictive_value(2)", "AUC(2)", "AUPRC(2)", "bias(2)"
]

# Create an empty DataFrame with the required columns
if not os.path.exists(csv_filename):
    df = pd.DataFrame(columns=["dir"] + columns)
    df.to_csv(csv_filename, index=False)



parent_dir = "/home/vatsal/MOSDAC/predictions/"  # Change this to your actual directory

# List all subdirectories that do not end with 'plots'
folders = [f for f in os.listdir(parent_dir) 
           if os.path.isdir(os.path.join(parent_dir, f)) and not f.endswith("plots") and not f.endswith(".npy")]


for folder in folders:
    address = os.path.join(parent_dir, folder)
    calculate_classification_scores(address, 21 ,16, csv_filename)



# import numpy as np
# import os
# import pandas as pd

# def calculate_classification_scores(dir, number_test_seq, seq_size):
#     np_files = [f for f in os.listdir(dir) if f.endswith('.npy')]
#     np_files.sort()

#     pred_file = np_files[number_test_seq:]
#     gt_file = np_files[:number_test_seq]

#     # Define metric lists
#     metrics = [
#         "false_alarm_rate", "critical_success_rate", "true_skill_statistics",
#         "prob_of_detection", "spec", "positive_predictive_value",
#         "negative_predictive_value", "AUC", "AUPRC", "bias"
#     ]

#     # Create empty lists for each metric (full, first half, second half)
#     metric_f = {m: [] for m in metrics}
#     metric_f1 = {f"{m}(1)": [] for m in metrics}
#     metric_f2 = {f"{m}(2)": [] for m in metrics}

#     for i in range(number_test_seq):
#         # Finding pred and gt file address
#         gt_add = os.path.join(dir, gt_file[i])
#         pred_add = os.path.join(dir, pred_file[i])

#         ground_truth = np.load(gt_add)
#         predicted = np.load(pred_add)

#         ground_truth = ground_truth[0]
#         predicted = predicted[0]

#         # Temporary lists for per-sequence scores
#         metric_a = {m: [] for m in metrics}

#         for index in range(ground_truth.shape[0]):
#             gt = ground_truth[index, :, :]
#             pred = predicted[index, :, :]

#             gt_binary = convert_to_binary(gt)
#             pred_binary = convert_to_binary(pred)

#             cs = ClassificationScores(gt_binary, pred_binary)

#             # Append individual scores to metric_a lists
#             metric_a["false_alarm_rate"].append(cs.false_alarm_rate())
#             metric_a["critical_success_rate"].append(cs.critical_success_rate())
#             metric_a["true_skill_statistics"].append(cs.true_skill_statistics())
#             metric_a["prob_of_detection"].append(cs.prob_of_detection())
#             metric_a["spec"].append(cs.spec())
#             metric_a["positive_predictive_value"].append(cs.positive_predictive_value())
#             metric_a["negative_predictive_value"].append(cs.negative_predictive_value())
#             metric_a["AUC"].append(cs.AUC())
#             metric_a["AUPRC"].append(cs.AUPRC())
#             metric_a["bias"].append(cs.bias())

#         # Compute mean values and update metric_f lists
#         for m in metrics:
#             metric_f[m].append(np.mean(metric_a[m]))
#             metric_f1[f"{m}(1)"].append(np.mean(metric_a[m][:seq_size // 2]))  # First half
#             metric_f2[f"{m}(2)"].append(np.mean(metric_a[m][seq_size // 2:]))  # Second half

#     # Combine all metrics into a DataFrame
#     results = {**metric_f, **metric_f1, **metric_f2}
#     results["dir"] = dir.split("/")[-2]  # Use directory name as the row index

#     # Convert to DataFrame
#     df = pd.DataFrame([results])

#     # Save results to CSV
#     csv_filename = "classification_scores.csv"
    
#     if os.path.exists(csv_filename):
#         df_existing = pd.read_csv(csv_filename)
#         df_existing = df_existing[df_existing["dir"] != results["dir"]]  # Remove old entry if exists
#         df = pd.concat([df_existing, df], ignore_index=True)

#     df.to_csv("results.csv", index=False)
#     print(f"Scores saved to {csv_filename}")




# calculate_classification_scores("/home/vatsal/MOSDAC/predictions/contrastive2_0.0001_relu_3", 8)



