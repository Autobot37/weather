
import numpy as np
import pandas as pd

def convert_to_binary_g_mean(data):

    positive_values = data[data > 0]  # Extract only positive values

    if positive_values.size > 0:  
        mean = np.mean(positive_values, axis=0)  # Compute mean safely
    else:
        mean = 0  
    data[data > mean] = 1
    data[data <= mean] = 0
    return data

def convert_to_binary_n_mean_1std(data):
    positive_values = data[data > 0]  # Extract only positive values

    if positive_values.size > 0:  
        mean = np.mean(positive_values, axis=0)  # Compute mean safely
        std = np.std(positive_values, axis = 0)
    else:
        mean = 0  
    data[data > (mean-std)] = 1
    data[data <= mean] = 0
    return data

class ClassificationScores:
    def __init__(self, binary_gt, binary_pred):
        pred_pos = (binary_pred == 1)
        gt_pos = (binary_gt == 1)
        pred_neg = (binary_pred == 0)
        gt_neg = (binary_gt == 0)

        self.true_positives = np.sum(pred_pos & gt_pos)
        self.true_negatives = np.sum(pred_neg & gt_neg)
        self.false_positives = np.sum(pred_pos & gt_neg)
        self.false_negatives = np.sum(pred_neg & gt_pos)

    # False Alarm Rate: Should be minimized
    def false_alarm_rate(self):
        denominator = self.true_negatives + self.false_positives
        return self.false_positives / denominator if denominator > 0 else 0  

    # Critical Success Rate
    def critical_success_rate(self):
        denominator = self.true_positives + self.false_negatives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0  

    # True Skill Statistics (TSS)
    def true_skill_statistics(self):
        denominator = (self.true_positives + self.false_negatives) * (self.true_negatives + self.false_positives)
        return ((self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)) / denominator if denominator > 0 else 0  

    # Probability of Detection (Recall)
    def prob_of_detection(self):
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0  

    # Specificity (True Negative Rate)
    def spec(self):
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / denominator if denominator > 0 else 0  

    # Precision (Positive Predictive Value)
    def positive_predictive_value(self):
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0  

    # Negative Predictive Value
    def negative_predictive_value(self):
        denominator = self.true_negatives + self.false_negatives
        return self.true_negatives / denominator if denominator > 0 else 0  

    # Area Under the ROC Curve (AUC)
    def AUC(self):
        return (self.prob_of_detection() + self.spec()) / 2  

    # Area Under Precision-Recall Curve (AUPRC)
    def AUPRC(self):
        precision = self.positive_predictive_value()
        recall = self.prob_of_detection()
        return (precision + recall) / 2  
    
    def bias(self):
        return (self.true_positives + self.false_positives)/(self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives)> 0 else 0
    



    def update_csv(dir_path, values, csv_filename ,columns_array):
        df = pd.read_csv(csv_filename)  # Load existing CSV
        dir_name = dir_path.split('/')[-2]  # Extract directory name

        # Check if the directory already exists in the CSV
        if dir_name in df["dir"].values:
            df.loc[df["dir"] == dir_name, columns_array] = values  # Update row
        else:
            new_row = {"dir": dir_name}
            new_row.update(dict(zip(columns_array, values)))  # Create row dictionary
            df = df.append(new_row, ignore_index=True)  # Append new row

        df.to_csv(csv_filename, index=False)




