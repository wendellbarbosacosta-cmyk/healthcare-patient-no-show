#!/usr/bin/env python3
"""
Pipeline de Análise e Previsão de No-Show Hospitalar
Autor: Portfólio de Ciência de Dados & Healthcare Analytics
Execução: python src/healthcare_pipeline.py --data data/healthcare_appointments_dataset.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Suporte seguro a UTF-8 no console Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def find_dataset_file(filepath: str) -> str:
    """Resolve o caminho do arquivo de dados em múltiplos locais comuns."""
    if os.path.exists(filepath):
        return filepath
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join("data", filepath),
        os.path.join(script_dir, "..", "data", os.path.basename(filepath)),
        os.path.join(script_dir, "..", filepath),
        os.path.join(script_dir, filepath),
        os.path.join("..", "data", os.path.basename(filepath)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return filepath

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Carrega o dataset e valida os dados de agendamentos clínicos."""
    resolved_path = find_dataset_file(filepath)
    print(f"[1/4] Carregando dados clínicos de '{resolved_path}'...")
    try:
        df = pd.read_csv(resolved_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filepath}' não foi encontrado.")
        sys.exit(1)

    total = len(df)
    no_shows = (df['attendance_status'] == 'No-Show (Faltou)').sum()
    rate = (no_shows / total) * 100
    total_loss = df['opportunity_cost'].sum()

    print(f"      Consultas Cadastradas: {total:,}")
    print(f"      Taxa Média de No-Show: {rate:.2f}%")
    print(f"      Prejuízo Total em Horas Ociosas: R$ {total_loss:,.2f}")
    print(f"      Lead Time Médio de Espera: {df['lead_time_days'].mean():.1f} dias")
    return df

def run_healthcare_pipeline(filepath: str):
    df = load_and_preprocess(filepath)

    print("\n[2/4] Indicadores por Especialidade Médica:")
    by_spec = df.groupby('specialty').agg(
        total_agendamentos=('appointment_id', 'count'),
        taxa_noshow=('attendance_status', lambda s: (s == 'No-Show (Faltou)').mean() * 100),
        lead_time_dias=('lead_time_days', 'mean'),
        perda_financeira=('opportunity_cost', 'sum')
    ).round(2).sort_values(by='taxa_noshow', ascending=False)
    print(by_spec.to_string())

    features = [
        'patient_age', 'chronic_condition', 'lead_time_days',
        'confirmed_digital', 'previous_no_shows', 'consultation_value'
    ]
    X = df[features]
    y = (df['attendance_status'] == 'No-Show (Faltou)').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Padronização de escala
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[3/4] Treinando modelos preditivos (Baseline vs Random Forest)...")

    # 1. Regressão Logística
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = lr_model.predict(X_test_scaled)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # 2. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=120, max_depth=6, random_state=42, class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_model.predict(X_test)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print("\n[4/4] Comparativo de Performance no Conjunto de Teste:")
    print("-" * 65)
    print(f"{'Métrica':<25} | {'Regressão Logística':<18} | {'Random Forest':<18}")
    print("-" * 65)
    print(f"{'ROC-AUC Score':<25} | {auc_lr:<18.4f} | {auc_rf:<18.4f}")
    print(f"{'Acurácia Global':<25} | {acc_lr*100:<17.1f}% | {acc_rf*100:<17.1f}%")
    print("-" * 65)

    print("\nRelatório de Classificação (Random Forest):\n")
    print(classification_report(y_test, y_pred_rf, target_names=['Compareceu', 'No-Show'], zero_division=0))

    print("Importância das Variáveis no Risco de No-Show:")
    for feat, imp in sorted(zip(features, rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  * {feat:<24}: {imp*100:.1f}%")

    print("\n[OK] Pipeline de Healthcare Analytics finalizado com sucesso!")
    return rf_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Previsão de No-Show Hospitalar")
    default_csv = "data/healthcare_appointments_dataset.csv"
    if not os.path.exists(default_csv) and os.path.exists("healthcare_appointments_dataset.csv"):
        default_csv = "healthcare_appointments_dataset.csv"
    parser.add_argument('--data', type=str, default=default_csv, help="Caminho do arquivo CSV de consultas")
    args = parser.parse_args()
    run_healthcare_pipeline(args.data)
