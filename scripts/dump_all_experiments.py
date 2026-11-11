import pandas as pd

df = pd.read_csv(r"C:\Ankshit\CVC-MAGNet-V2\reports\master_ablation_ledger.csv")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

for idx, row in df.iterrows():
    exp_id = row.get("exp_id") if pd.notna(row.get("exp_id")) else row.get("experiment_id")
    stage = row.get("stage")
    val_auroc = row.get("val_macro_auroc") if pd.notna(row.get("val_macro_auroc")) else row.get("Val_Macro_AUROC_2558")
    norm_auroc = row.get("Normal_AUROC") if pd.notna(row.get("Normal_AUROC")) else row.get("val_normal_auroc")
    bord_auroc = row.get("Borderline_AUROC") if pd.notna(row.get("Borderline_AUROC")) else row.get("val_borderline_auroc")
    abn_auroc = row.get("Abnormal_AUROC") if pd.notna(row.get("Abnormal_AUROC")) else row.get("val_abnormal_auroc")
    untraced = row.get("Untraced_1780_AUROC") if pd.notna(row.get("Untraced_1780_AUROC")) else row.get("untraced_auroc")
    desc = row.get("description", "")
    
    print(f"{idx+1:02d} | {str(exp_id):<42} | Stg {str(stage):<2} | Macro: {float(val_auroc):.4f} | N:{float(norm_auroc):.4f} B:{float(bord_auroc):.4f} A:{float(abn_auroc):.4f} | Untr:{float(untraced):.4f} | {desc}")

