"""
Deloitte Forensic Technology - Financial Ledger Anomaly Detection & Benford's Law Audit
Author: Prashant Kumar
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def generate_financial_ledger(n_transactions=6000):
    np.random.seed(42)
    
    # Authentic corporate accounting transactions follow log-normal distribution
    amounts = np.random.lognormal(mean=5.5, sigma=1.2, size=n_transactions)
    
    # Inject suspicious fraudulent/erroneous entries (round amounts, sudden spikes)
    n_fraud = int(n_transactions * 0.03)
    fraud_indices = np.random.choice(n_transactions, n_fraud, replace=False)
    amounts[fraud_indices] = np.random.choice([9999.0, 4999.0, 85000.0, 120000.0, 7500.0], n_fraud)
    
    df = pd.DataFrame({
        'TransactionID': [f"TXN_AU_{i:06d}" for i in range(1, n_transactions + 1)],
        'Amount': amounts,
        'VendorID': np.random.choice([f"VEND_{k:03d}" for k in range(1, 150)], n_transactions),
        'ApproverLevel': np.random.choice([1, 2, 3, 4], n_transactions, p=[0.4, 0.3, 0.2, 0.1]),
        'IsWeekend': np.random.choice([0, 1], n_transactions, p=[0.85, 0.15])
    })
    return df

def audit_benford_law(amounts):
    first_digits = [int(str(f"{a:.2f}").replace('.', '').lstrip('0')[0]) for a in amounts if a > 0]
    series = pd.Series(first_digits)
    freq = (series.value_counts(normalize=True).sort_index() * 100).to_dict()
    
    # Theoretical Benford's Law percentages
    benford_expected = {d: np.log10(1 + 1/d) * 100 for d in range(1, 10)}
    return freq, benford_expected

def run_forensic_investigation():
    print("[1/3] Generating Corporate General Ledger Dataset...")
    df = generate_financial_ledger(6000)
    
    print("[2/3] Performing Benford's Law Mathematical First-Digit Audit...")
    actual_dist, expected_dist = audit_benford_law(df['Amount'])
    print("  Digit | Actual % | Benford Expected % | Variance")
    print("  ---------------------------------------------")
    for d in range(1, 10):
        act = actual_dist.get(d, 0.0)
        exp = expected_dist[d]
        var = act - exp
        print(f"    {d}   |  {act:5.2f}%  |      {exp:5.2f}%       | {var:+5.2f}%")
        
    print("
[3/3] Training Isolation Forest Unsupervised Anomaly Detector...")
    X = df[['Amount', 'ApproverLevel', 'IsWeekend']]
    iso = IsolationForest(contamination=0.035, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(X)
    anomalies = df[df['Anomaly_Flag'] == -1]
    
    print(f"  Flagged Suspicious Transactions: {len(anomalies)} / {len(df)}")
    print("
Sample High-Risk Flagged Ledger Transactions:")
    print(anomalies[['TransactionID', 'Amount', 'VendorID', 'ApproverLevel']].head(6).to_string(index=False))

if __name__ == '__main__':
    run_forensic_investigation()
