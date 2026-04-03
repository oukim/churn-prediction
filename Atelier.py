# =============================================================================
# Atelier N° 05 : Prédiction du Churn Client avec Python
# Université Sultan Moulay Slimane – Master MD3SI – S1
# Module : Python Avancé (M113) | Pr. Siham BAKKOURI
# Année universitaire : 2025/2026
# =============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_auc_score,
                              roc_curve, ConfusionMatrixDisplay)

# =============================================================================
# PARTIE 1 : EXPLORATION DU DATASET
# =============================================================================
print("\n>>>>>> PARTIE 1 : Exploration du dataset")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(f"\n — Lignes x Colonnes : {df.shape[0]} x {df.shape[1]}")
print(f"\n — Variables :\n{list(df.columns)}")
print(f"\n — Types des variables :\n{df.dtypes}")
print(f"\n — Variable cible : Churn")
print(f"     Modalités : {df['Churn'].unique()}")
print(f"   Distribution :\n{df['Churn'].value_counts()}")

print(f"\n — Valeurs manquantes :")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0
      else "Aucune valeur NaN détectée.")

print(f"\n Aperçu des 5 premières lignes :", df.head())


# =============================================================================
# PARTIE 2 : NETTOYAGE DES DONNÉES
# =============================================================================
print("\n>>> PARTIE 2 : Nettoyage des données")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print(f"\nQ1 — Valeurs manquantes après conversion numérique :\n",
      df.isnull().sum()[df.isnull().sum() > 0])

n_miss = df['TotalCharges'].isnull().sum()
print(f"\nQ2 — {n_miss} valeurs manquantes dans TotalCharges ({n_miss/len(df)*100:.2f}%)")
print("     Decision : suppression (proportion < 0.5%, impact négligeable)")
df.dropna(subset=['TotalCharges'], inplace=True)
print(f"     Dataset après suppression : {df.shape[0]} lignes")

df.drop(columns=['customerID'], inplace=True)

print(f"\nQ3 — Transformation des variables catégorielles :")
print("     Les algorithmes ML travaillent uniquement avec des nombres.")
print("     Exemple : 'Yes'/'No' → 1/0 | 'Month-to-month' → colonne binaire")

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

cols_binary = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in cols_binary:
    df[col] = LabelEncoder().fit_transform(df[col])

cols_ohe = ['MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=cols_ohe, drop_first=True)

print(f"\n     Dataset final après encodage : {df.shape[1]} colonnes")


# =============================================================================
# PARTIE 3 : COMPRÉHENSION DES DONNÉES
# =============================================================================
print("\n>>> PARTIE 3 : Compréhension des données")

cols_num = ['tenure', 'MonthlyCharges', 'TotalCharges']
print(f"\nStatistiques descriptives :\n{df[cols_num].describe().round(2)}")
print(f"\n — Moyenne MonthlyCharges : {df['MonthlyCharges'].mean():.2f} $")

stds = df[cols_num].std()
print(f"\n — Écarts-types :\n{stds.round(2)}")
print(f"     Variable avec la plus grande dispersion : {stds.idxmax()} (σ={stds.max():.2f})")

vc  = df['Churn'].value_counts()
vcp = df['Churn'].value_counts(normalize=True) * 100
print(f"\n — Distribution Churn :")
print(f"     No  (0) : {vc[0]} clients ({vcp[0]:.1f}%)")
print(f"     Yes (1) : {vc[1]} clients ({vcp[1]:.1f}%)")
print(f"     ⚠ Dataset déséquilibré → privilégier Recall et F1-score")

# Graphique P3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Partie 3 — Distribution des variables clés", fontsize=13, fontweight='bold')

axes[0].bar(['No Churn', 'Churn'], vc.values,
            color=['#2196F3', '#F44336'], edgecolor='white')
axes[0].set_title("Distribution Churn")
axes[0].set_ylabel("Nombre de clients")
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 30, f'{v}\n({vcp.values[i]:.1f}%)', ha='center', fontsize=10)

axes[1].hist(df[df['Churn']==0]['MonthlyCharges'], bins=30,
             alpha=0.6, color='#2196F3', label='No Churn')
axes[1].hist(df[df['Churn']==1]['MonthlyCharges'], bins=30,
             alpha=0.6, color='#F44336', label='Churn')
axes[1].set_title("MonthlyCharges selon Churn")
axes[1].set_xlabel("Frais mensuels ($)")
axes[1].set_ylabel("Nombre de clients")
axes[1].legend()

axes[2].hist(df[df['Churn']==0]['tenure'], bins=30,
             alpha=0.6, color='#2196F3', label='No Churn')
axes[2].hist(df[df['Churn']==1]['tenure'], bins=30,
             alpha=0.6, color='#F44336', label='Churn')
axes[2].set_title("Ancienneté (tenure) selon Churn")
axes[2].set_xlabel("Ancienneté (mois)")
axes[2].set_ylabel("Nombre de clients")
axes[2].legend()

plt.tight_layout()
plt.savefig("P3_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n     [Graphique sauvegardé : P3_distributions.png]")


# =============================================================================
# PARTIE 4 : ANALYSE DESCRIPTIVE
# =============================================================================
print("\n>>> PARTIE 4 : Analyse descriptive")

t_churn   = df[df['Churn']==1]['tenure']
t_nochurn = df[df['Churn']==0]['tenure']
print(f"\nQ1 — Ancienneté moyenne :")
print(f"     Clients qui ont churné  : {t_churn.mean():.1f} mois")
print(f"     Clients qui sont restés : {t_nochurn.mean():.1f} mois")
print("     → Les clients récents (faible tenure) quittent beaucoup plus souvent.")

mc_c  = df[df['Churn']==1]['MonthlyCharges'].mean()
mc_nc = df[df['Churn']==0]['MonthlyCharges'].mean()
print(f"\nQ2 — Charges mensuelles moyennes :")
print(f"     Clients churned : {mc_c:.2f} $")
print(f"     Clients restés  : {mc_nc:.2f} $")
print("     → Les churners paient en moyenne plus cher.")

senior_churn    = df[df['SeniorCitizen']==1]['Churn'].mean() * 100
nonsenior_churn = df[df['SeniorCitizen']==0]['Churn'].mean() * 100
print(f"\nQ3 — Profils à risque :")
print(f"     Seniors     → taux de churn : {senior_churn:.1f}%")
print(f"     Non seniors → taux de churn : {nonsenior_churn:.1f}%")
print("     Profils à risque identifiés :")
print("       • Contrat mensuel (Month-to-month)")
print("       • Faible ancienneté (< 12 mois)")
print("       • Frais mensuels élevés (> 70 $)")
print("       • Clients seniors")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Partie 4 — Analyse descriptive", fontsize=14, fontweight='bold', y=1.02)

# Couleurs
color_no  = '#2196F3'   # bleu  → No Churn
color_yes = '#F44336'   # rouge → Churn

# ── Données séparées ─────────────────────────────────────────────
tenure_no  = df[df['Churn'] == 0]['tenure']
tenure_yes = df[df['Churn'] == 1]['tenure']
mc_no      = df[df['Churn'] == 0]['MonthlyCharges']
mc_yes     = df[df['Churn'] == 1]['MonthlyCharges']

# ── Graphique 1 : Tenure ─────────────────────────────────────────
bp1 = axes[0].boxplot(
    [tenure_no, tenure_yes],
    labels=['No Churn', 'Churn'],
    patch_artist=True,          # remplissage coloré
    widths=0.5,
    medianprops=dict(color='white', linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker='o', markersize=3, alpha=0.4)
)
bp1['boxes'][0].set_facecolor(color_no)
bp1['boxes'][0].set_alpha(0.75)
bp1['boxes'][1].set_facecolor(color_yes)
bp1['boxes'][1].set_alpha(0.75)

# Ligne de la moyenne
axes[0].axhline(tenure_no.mean(),  color=color_no,  linestyle='--',
                linewidth=1.2, alpha=0.7, label=f'Moy. No Churn ({tenure_no.mean():.1f} mois)')
axes[0].axhline(tenure_yes.mean(), color=color_yes, linestyle='--',
                linewidth=1.2, alpha=0.7, label=f'Moy. Churn ({tenure_yes.mean():.1f} mois)')

axes[0].set_title("Ancienneté selon le Churn", fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel("Statut client", fontsize=10)
axes[0].set_ylabel("Tenure (mois)", fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
axes[0].set_facecolor('#FAFAFA')

# ── Graphique 2 : MonthlyCharges ─────────────────────────────────
bp2 = axes[1].boxplot(
    [mc_no, mc_yes],
    labels=['No Churn', 'Churn'],
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color='white', linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker='o', markersize=3, alpha=0.4)
)
bp2['boxes'][0].set_facecolor(color_no)
bp2['boxes'][0].set_alpha(0.75)
bp2['boxes'][1].set_facecolor(color_yes)
bp2['boxes'][1].set_alpha(0.75)

# Ligne de la moyenne
axes[1].axhline(mc_no.mean(),  color=color_no,  linestyle='--',
                linewidth=1.2, alpha=0.7, label=f'Moy. No Churn ({mc_no.mean():.1f} $)')
axes[1].axhline(mc_yes.mean(), color=color_yes, linestyle='--',
                linewidth=1.2, alpha=0.7, label=f'Moy. Churn ({mc_yes.mean():.1f} $)')

axes[1].set_title("Frais mensuels selon le Churn", fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel("Statut client", fontsize=10)
axes[1].set_ylabel("MonthlyCharges ($)", fontsize=10)
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
axes[1].set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig("P4_analyse_descriptive.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Graphique sauvegardé : P4_analyse_descriptive.png")


# =============================================================================
# PARTIE 5 : MODÉLISATION MACHINE LEARNING
# =============================================================================
print("\n>>> PARTIE 5 : Modélisation Machine Learning")

X = df.drop('Churn', axis=1)
y = df['Churn']
print(f"\n     X : {X.shape} | y : {y.shape}")

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_s, y_train)
y_pred_lr    = model_lr.predict(X_test_s)
y_proba_lr   = model_lr.predict_proba(X_test_s)[:, 1]
print("\n     [✓] Régression Logistique entraînée")

model_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
model_tree.fit(X_train, y_train)
y_pred_tree  = model_tree.predict(X_test)
y_proba_tree = model_tree.predict_proba(X_test)[:, 1]
print("     [✓] Arbre de Décision entraîné (max_depth=5)")

model_knn = KNeighborsClassifier(n_neighbors=7)
model_knn.fit(X_train_s, y_train)
y_pred_knn   = model_knn.predict(X_test_s)
y_proba_knn  = model_knn.predict_proba(X_test_s)[:, 1]
print("     [✓] KNN entraîné (k=7)")


# =============================================================================
# PARTIE 7 : ÉVALUATION DES MODÈLES
# =============================================================================
print("\n>>> PARTIE 7 : Évaluation des modèles")

def evaluer(nom, y_true, y_pred, y_proba):
    rep = classification_report(y_true, y_pred, output_dict=True)
    return {
        'Modèle':    nom,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(rep['1']['precision'], 4),
        'Recall':    round(rep['1']['recall'], 4),
        'F1-score':  round(rep['1']['f1-score'], 4),
        'AUC-ROC':   round(roc_auc_score(y_true, y_proba), 4),
        'y_pred':    y_pred,
        'y_proba':   y_proba
    }

results = [
    evaluer("Régression Logistique", y_test, y_pred_lr,   y_proba_lr),
    evaluer("Arbre de Décision",     y_test, y_pred_tree,  y_proba_tree),
    evaluer("KNN (k=7)",             y_test, y_pred_knn,   y_proba_knn),
]

# Q1 — Accuracy
print(f"\nQ1 — Accuracy des modèles :")
for r in results:
    print(f"     {r['Modèle']:30s} : {r['Accuracy']*100:.2f}%")

# Q2 — Métrique la plus pertinente
print(f"\nQ2 — Métrique la plus pertinente : RECALL")
print("     Raison : manquer un churner (faux négatif) est plus coûteux")
print("     que cibler un client qui reste (faux positif).")
print("     Le F1-score est aussi pertinent car il équilibre Precision/Recall.")

# Q3 — Recall pour chaque modèle
print(f"\nQ3 — Recall pour la détection des churners :")
for r in results:
    qualite = "Bon" if r['Recall'] >= 0.65 else "Moyen" if r['Recall'] >= 0.50 else "Faible"
    print(f"     {r['Modèle']:30s} : Recall={r['Recall']:.4f} [{qualite}]")

# Tableau comparatif
df_res = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ['y_pred', 'y_proba']} for r in results])
print(f"\n     Tableau comparatif :")
print(df_res.to_string(index=False))

# ── SÉLECTION DU MEILLEUR MODÈLE SELON LE RECALL ─────────────────────────────
print("\n" + "="*60)

# Trouver le modèle avec le Recall le plus élevé sur la classe Churn (1)
best_model = max(results, key=lambda x: x['Recall'])

print(f"\n🏆 Meilleur modèle (selon Recall) : {best_model['Modèle']}")
print(f"\n   Pourquoi le Recall ?")
print(f"   Dans le contexte du churn, manquer un client qui part (Faux Négatif)")
print(f"   coûte bien plus cher à l'entreprise que cibler un client qui reste.")
print(f"   Le Recall mesure exactement la capacité du modèle à détecter")
print(f"   les vrais churners → c'est la métrique prioritaire ici.")

print(f"\n   Comparaison des Recall :")
for r in sorted(results, key=lambda x: x['Recall'], reverse=True):
    marker = " ◀ MEILLEUR" if r['Modèle'] == best_model['Modèle'] else ""
    bar    = "█" * int(r['Recall'] * 30)
    print(f"   {r['Modèle']:30s} : {r['Recall']:.4f}  {bar}{marker}")

# Rapport détaillé du meilleur modèle
print(f"\n   Rapport détaillé — {best_model['Modèle']} :")
print(classification_report(
    y_test,
    best_model['y_pred'],
    target_names=['No Churn', 'Churn']
))

# Interprétation chiffrée du Recall
n_vrais_churners = (y_test == 1).sum()
n_detectes       = int(best_model['Recall'] * n_vrais_churners)
n_rates          = n_vrais_churners - n_detectes

print(f"   Sur {n_vrais_churners} vrais churners dans le test set :")
print(f"   ✅ Détectés correctement : {n_detectes} clients")
print(f"   ❌ Manqués (non détectés) : {n_rates} clients")
print(f"   → Chaque client manqué = revenus perdus pour l'entreprise")

print("="*60)
# ─────────────────────────────────────────────────────────────────────────────

# Rapport Régression Logistique (meilleur AUC-ROC global)
print(f"\n     Rapport Régression Logistique (meilleur AUC-ROC global) :")
print(classification_report(y_test, y_pred_lr, target_names=['No Churn', 'Churn']))

# Graphique P7 — Matrices de confusion + Courbes ROC
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Partie 7 — Évaluation des modèles", fontsize=14, fontweight='bold')

for i, r in enumerate(results):
    # Surligner le meilleur modèle
    titre = r['Modèle']
    if r['Modèle'] == best_model['Modèle']:
        titre += "  ★ Meilleur Recall"

    cm = confusion_matrix(y_test, r['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=axes[0, i], colorbar=False, cmap='Blues')
    axes[0, i].set_title(titre, fontsize=10,
                          fontweight='bold' if r['Modèle'] == best_model['Modèle'] else 'normal')

    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    axes[1, i].plot(fpr, tpr, color='#2196F3', lw=2,
                    label=f"AUC = {r['AUC-ROC']:.3f}")
    axes[1, i].plot([0, 1], [0, 1], '--', color='gray', lw=1)
    axes[1, i].fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    axes[1, i].set_title(f"ROC — {r['Modèle']}", fontsize=10)
    axes[1, i].set_xlabel("Taux Faux Positifs")
    axes[1, i].set_ylabel("Recall")
    axes[1, i].legend()

plt.tight_layout()
plt.savefig("P7_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("     [Graphique sauvegardé : P7_evaluation.png]")

# Graphique comparaison métriques
fig, ax = plt.subplots(figsize=(11, 5))
metriques = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']
colors_m  = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0']
x         = np.arange(len(df_res))
width     = 0.15

for j, (m, c) in enumerate(zip(metriques, colors_m)):
    ax.bar(x + j * width, df_res[m], width, label=m, color=c, alpha=0.85)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(df_res['Modèle'], fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Comparaison des métriques — 3 modèles")
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax.legend()
plt.tight_layout()
plt.savefig("P7_metriques.png", dpi=150, bbox_inches='tight')
plt.close()
print("     [Graphique sauvegardé : P7_metriques.png]")


# =============================================================================
# PARTIE 8 : INTERPRÉTATION MÉTIER
# =============================================================================
print("\n>>> PARTIE 8 : Interprétation métier")

importances = pd.Series(model_tree.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)

plt.figure(figsize=(10, 6))
top10.sort_values().plot(kind='barh', color='#2196F3', edgecolor='white')
plt.title("Top 10 — Features les plus importantes (Arbre de Décision)",
          fontsize=12, fontweight='bold')
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("P8_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n     [Graphique sauvegardé : P8_feature_importance.png]")

print(f"\nQ1 — Principaux facteurs du churn :")
print(f"     {list(top10.index)}")

print(f"\nQ2 — Profils clients les plus à risque :")
print("     • Contrat mensuel (Month-to-month)")
print("     • Faible ancienneté (< 12 mois)")
print("     • Frais mensuels élevés (> 70 $)")
print("     • Clients seniors (SeniorCitizen = 1)")
print("     • Sans partenaire ni personnes à charge")

print(f"\nQ3 — Actions pour réduire le churn :")
print("     • Proposer des réductions pour les contrats annuels ou 2 ans")
print("     • Bonus de fidélité lors des 6 premiers mois")
print("     • Offres personnalisées pour les clients à hauts frais mensuels")
print("     • Programme support dédié pour les seniors")

print(f"\nQ4 — Amélioration de la fidélisation :")
print("     • Scoring mensuel automatique pour chaque client")
print("     • Alertes si probabilité de churn > 60%")
print("     • Segmentation client par profil de risque")

print(f"\nQ5 — Exploitation en production :")
print("     • Sauvegarder le modèle : joblib.dump(model_lr, 'churn_model.pkl')")
print("     • API Flask/FastAPI : endpoint /predict retournant P(churn)")
print("     • Intégration CRM pour alerter les agents commerciaux")
print("     • Réentraînement trimestriel avec les nouvelles données")

# Démonstration — Prédiction nouveau client
print(f"\n--- Démonstration : prédiction pour un nouveau client ---")
nc = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
nc['tenure']         = 2
nc['MonthlyCharges'] = 85.0
nc['TotalCharges']   = 170.0
nc['SeniorCitizen']  = 0
nc_s  = scaler.transform(nc)
prob  = model_lr.predict_proba(nc_s)[0][1]
print(f"     Profil : tenure=2 mois | charges=85$/mois")
print(f"     P(churn) = {prob*100:.1f}%")
print(f"     Decision : {'ACTION REQUISE ⚠' if prob > 0.5 else 'Faible risque ✓'}")

# Résumé final
print(f"\n{'='*60}")
print(f"  MEILLEUR MODÈLE (Recall)  : {best_model['Modèle']}")
print(f"  Recall    : {best_model['Recall']*100:.2f}%")
print(f"  Precision : {best_model['Precision']*100:.2f}%")
print(f"  F1-score  : {best_model['F1-score']*100:.2f}%")
print(f"  Accuracy  : {best_model['Accuracy']*100:.2f}%")
print(f"  AUC-ROC   : {best_model['AUC-ROC']:.4f}")
print(f"{'='*60}")
print("\nAtelier 05 terminé. Graphiques générés :")
print("  P3_distributions.png     | P4_analyse_descriptive.png")
print("  P7_evaluation.png        | P7_metriques.png")
print("  P8_feature_importance.png")