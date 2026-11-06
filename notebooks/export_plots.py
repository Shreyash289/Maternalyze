import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
plt.rcParams['figure.dpi'] = 150

OUT = Path('../notebooks/eda_plots')
OUT.mkdir(exist_ok=True)

for candidate in [Path('../data/raw/gdm_dataset.csv'), Path('data/raw/gdm_dataset.csv')]:
    if candidate.exists():
        DATA_PATH = candidate
        break

df_raw = pd.read_csv(DATA_PATH, low_memory=False)
df = df_raw.copy()

for col in ['Glucoselevel0minblood', 'Glucoselevel120minblood', 'Glucoselevelblood', 'VitaminDlevelblood']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['GDM']  = df['Gestational Diabetes'].str.strip().map({'Yes': 1, 'No': 0})
df['SCBU'] = df['SCBU admission (Yes/NO)'].str.strip().map({'Yes': 1, 'No': 0})
gdm_df = df.dropna(subset=['GDM'])

NUM_COLS = [
    'AgeAtStartOfSpell', 'WeightMeasured', 'Height', 'Body Mass Index at Booking',
    'Glucoselevel0minblood', 'Glucoselevel120minblood',
    'SystolicBloodPressureCuff', 'Diastolic Blood Pressure',
    'Gestation', 'BabyBirthWeight', 'BW Centile',
    'EstimatedTotalBloodLoss', 'LOS mother after delivery',
    'TotalApgarScoreat1minutes', 'APGAR_Score_5', 'TotalApgarScoreat10minutes'
]
LABELS = {
    'AgeAtStartOfSpell': 'Age (years)',
    'WeightMeasured': 'Weight (kg)',
    'Height': 'Height (cm)',
    'Body Mass Index at Booking': 'BMI at Booking',
    'Glucoselevel0minblood': 'OGTT 0-min Glucose (mmol/L)',
    'Glucoselevel120minblood': 'OGTT 120-min Glucose (mmol/L)',
    'SystolicBloodPressureCuff': 'Systolic BP (mmHg)',
    'Diastolic Blood Pressure': 'Diastolic BP (mmHg)',
    'Gestation': 'Gestation (weeks)',
    'BabyBirthWeight': 'Birth Weight (g)',
    'BW Centile': 'Birth Weight Centile',
    'EstimatedTotalBloodLoss': 'Estimated Blood Loss (mL)',
    'LOS mother after delivery': 'Length of Stay (days)',
    'TotalApgarScoreat1minutes': 'APGAR @ 1 min',
    'APGAR_Score_5': 'APGAR @ 5 min',
    'TotalApgarScoreat10minutes': 'APGAR @ 10 min'
}

def save(fig, name):
    path = OUT / f'{name}.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  saved: {path.name}')

# ── 1. Mean-Median summary table ────────────────────────────────────────────
print('1. Mean-Median table...')
stats = df[NUM_COLS].agg(['count','mean','median','std','min','max']).T
stats.index = [LABELS.get(c, c) for c in stats.index]
stats = stats.round(2)
stats.columns = ['N', 'Mean', 'Median', 'Std', 'Min', 'Max']
stats['Mean-Median Gap'] = (stats['Mean'] - stats['Median']).round(2)

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')
tbl = ax.table(
    cellText=stats.reset_index().values,
    colLabels=['Feature'] + list(stats.columns),
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f2f2f2')
ax.set_title('Mean & Median Summary — GDM Dataset', fontsize=13, fontweight='bold', pad=12)
save(fig, '01_mean_median_summary')

# ── 2. Histograms ────────────────────────────────────────────────────────────
print('2. Histograms...')
n, ncols = len(NUM_COLS), 4
nrows = -(-n // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=40, color='steelblue', edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.axvline(data.mean(),   color='crimson',    linestyle='--', linewidth=1.4, label=f'Mean {data.mean():.1f}')
    ax.axvline(data.median(), color='darkorange', linestyle='-',  linewidth=1.4, label=f'Median {data.median():.1f}')
    ax.set_title(LABELS.get(col, col), fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=8)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Histograms — GDM Dataset  (red = mean | orange = median)', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '02_histograms')

# ── 3. Density plots (KDE) ───────────────────────────────────────────────────
print('3. Density plots...')
KDE_COLS = [
    'AgeAtStartOfSpell', 'Body Mass Index at Booking', 'WeightMeasured',
    'Glucoselevel0minblood', 'Glucoselevel120minblood',
    'SystolicBloodPressureCuff', 'Diastolic Blood Pressure',
    'Gestation', 'BabyBirthWeight', 'BW Centile',
    'EstimatedTotalBloodLoss', 'LOS mother after delivery'
]
nrows = -(-len(KDE_COLS) // 4)
fig, axes = plt.subplots(nrows, 4, figsize=(18, nrows * 3.5))
axes = axes.flatten()
for i, col in enumerate(KDE_COLS):
    ax = axes[i]
    for gdm_val, label, color in [(0, 'No GDM', '#4c78a8'), (1, 'GDM', '#e45756')]:
        subset = gdm_df.loc[gdm_df['GDM'] == gdm_val, col].dropna()
        if len(subset) > 10:
            subset.plot.kde(ax=ax, label=label, color=color, linewidth=2)
            ax.fill_between(ax.lines[-1].get_xdata(), ax.lines[-1].get_ydata(), alpha=0.12, color=color)
    ax.set_title(LABELS.get(col, col), fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Density Plots (KDE) — Blue: No GDM  |  Red: GDM', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '03_density_plots_kde')

# ── 4. Box plots ─────────────────────────────────────────────────────────────
print('4. Box plots...')
BOX_COLS = [
    'AgeAtStartOfSpell', 'Body Mass Index at Booking', 'WeightMeasured',
    'Glucoselevel0minblood', 'Glucoselevel120minblood',
    'SystolicBloodPressureCuff', 'Diastolic Blood Pressure',
    'BabyBirthWeight', 'Gestation', 'LOS mother after delivery'
]
nrows = -(-len(BOX_COLS) // 2)
fig, axes = plt.subplots(nrows, 2, figsize=(14, nrows * 3.8))
axes = axes.flatten()
for i, col in enumerate(BOX_COLS):
    ax = axes[i]
    plot_df = gdm_df[['GDM', col]].dropna().copy()
    plot_df['GDM Label'] = plot_df['GDM'].map({0: 'No GDM', 1: 'GDM'})
    sns.boxplot(
        data=plot_df, x='GDM Label', y=col, ax=ax,
        palette={'No GDM': '#4c78a8', 'GDM': '#e45756'},
        width=0.45, flierprops=dict(marker='o', markersize=2, alpha=0.3),
        order=['No GDM', 'GDM']
    )
    ax.set_title(LABELS.get(col, col), fontsize=9, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(labelsize=8)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Box Plots by GDM Status', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '04_box_plots')

# ── 5. Scatter plots ─────────────────────────────────────────────────────────
print('5. Scatter plots...')
from matplotlib.lines import Line2D
SCATTER_PAIRS = [
    ('Glucoselevel0minblood',     'Glucoselevel120minblood',    'OGTT: 0-min vs 120-min Glucose'),
    ('Body Mass Index at Booking','Glucoselevel0minblood',      'BMI vs OGTT 0-min Glucose'),
    ('Body Mass Index at Booking','Glucoselevel120minblood',    'BMI vs OGTT 120-min Glucose'),
    ('AgeAtStartOfSpell',         'Body Mass Index at Booking', 'Age vs BMI'),
    ('Gestation',                 'BabyBirthWeight',            'Gestation vs Birth Weight'),
    ('SystolicBloodPressureCuff', 'Diastolic Blood Pressure',   'Systolic vs Diastolic BP'),
    ('BW Centile',                'BabyBirthWeight',            'Birth Weight Centile vs Birth Weight (g)'),
    ('AgeAtStartOfSpell',         'Glucoselevel120minblood',    'Age vs OGTT 120-min Glucose'),
]
nrows = -(-len(SCATTER_PAIRS) // 2)
fig, axes = plt.subplots(nrows, 2, figsize=(14, nrows * 4))
axes = axes.flatten()
for i, (xcol, ycol, title) in enumerate(SCATTER_PAIRS):
    ax = axes[i]
    plot_df = gdm_df[[xcol, ycol, 'GDM']].dropna()
    sample = plot_df.sample(min(3000, len(plot_df)), random_state=42)
    ax.scatter(sample[xcol], sample[ycol],
               c=sample['GDM'].map({0: '#4c78a8', 1: '#e45756'}),
               alpha=0.35, s=12, linewidths=0)
    clean = plot_df[[xcol, ycol]].dropna()
    if len(clean) > 10:
        m, b = np.polyfit(clean[xcol], clean[ycol], 1)
        xline = np.linspace(clean[xcol].quantile(0.01), clean[xcol].quantile(0.99), 100)
        ax.plot(xline, m*xline+b, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
    legend_els = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#4c78a8', markersize=7, label='No GDM'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#e45756', markersize=7, label='GDM'),
    ]
    ax.legend(handles=legend_els, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel(LABELS.get(xcol, xcol), fontsize=8)
    ax.set_ylabel(LABELS.get(ycol, ycol), fontsize=8)
    ax.tick_params(labelsize=8)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Scatter Plots — Blue: No GDM  |  Red: GDM  |  Dashed: trend line', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '05_scatter_plots')

# ── 6. Correlation heatmap ───────────────────────────────────────────────────
print('6. Correlation heatmap...')
CORR_COLS = [
    'AgeAtStartOfSpell', 'Body Mass Index at Booking', 'WeightMeasured', 'Height',
    'Glucoselevel0minblood', 'Glucoselevel120minblood',
    'SystolicBloodPressureCuff', 'Diastolic Blood Pressure',
    'Gestation', 'BabyBirthWeight', 'BW Centile',
    'EstimatedTotalBloodLoss', 'LOS mother after delivery', 'GDM', 'SCBU'
]
corr = df[CORR_COLS].corr()
corr.index   = [LABELS.get(c, c) for c in corr.index]
corr.columns = [LABELS.get(c, c) for c in corr.columns]
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.4, annot_kws={'size': 7}, ax=ax)
ax.set_title('Pearson Correlation — Key Numeric Features', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '06_correlation_heatmap')

# ── 7. GDM prevalence by ethnicity & obesity ─────────────────────────────────
print('7. GDM by ethnicity & obesity...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
eth_gdm = (gdm_df.groupby('Ethnicity')['GDM']
           .agg(['mean','count']).query('count >= 100')
           .sort_values('mean', ascending=True))
eth_gdm['mean_pct'] = eth_gdm['mean'] * 100
ax = axes[0]
bars = ax.barh(eth_gdm.index, eth_gdm['mean_pct'], color='steelblue', edgecolor='white')
for bar, (_, row) in zip(bars, eth_gdm.iterrows()):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f"{row['mean_pct']:.1f}% (n={int(row['count'])})", va='center', fontsize=8)
ax.set_xlabel('GDM Rate (%)'); ax.set_title('GDM Prevalence by Ethnicity', fontweight='bold')
ax.set_xlim(0, eth_gdm['mean_pct'].max()*1.35)

ax2 = axes[1]
obese_gdm = (gdm_df.dropna(subset=['Obese?']).groupby('Obese?')['GDM'].agg(['mean','count']))
obese_gdm['mean_pct'] = obese_gdm['mean'] * 100
bars2 = ax2.bar(obese_gdm.index, obese_gdm['mean_pct'], color=['#4c78a8','#e45756'], edgecolor='white', width=0.4)
for bar, (_, row) in zip(bars2, obese_gdm.iterrows()):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{row['mean_pct']:.1f}%\n(n={int(row['count'])})", ha='center', fontsize=9)
ax2.set_xlabel('Obese at Booking?'); ax2.set_ylabel('GDM Rate (%)')
ax2.set_title('GDM Prevalence: Obese vs Non-Obese', fontweight='bold')
ax2.set_ylim(0, obese_gdm['mean_pct'].max()*1.3)
plt.tight_layout()
save(fig, '07_gdm_by_ethnicity_obesity')

# ── 8. APGAR distributions ───────────────────────────────────────────────────
print('8. APGAR distributions...')
APGAR_COLS   = ['TotalApgarScoreat1minutes', 'APGAR_Score_5', 'TotalApgarScoreat10minutes']
APGAR_LABELS = ['APGAR @ 1 min', 'APGAR @ 5 min', 'APGAR @ 10 min']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, label in zip(axes, APGAR_COLS, APGAR_LABELS):
    data = df[col].dropna()
    counts = data.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='mediumseagreen', edgecolor='white', width=0.7)
    ax.axvline(data.mean(),   color='crimson',    linestyle='--', linewidth=1.5, label=f'Mean {data.mean():.2f}')
    ax.axvline(data.median(), color='darkorange', linestyle='-',  linewidth=1.5, label=f'Median {data.median():.0f}')
    ax.set_title(label, fontweight='bold'); ax.set_xlabel('Score'); ax.set_ylabel('Count')
    ax.legend(fontsize=8)
fig.suptitle('APGAR Score Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '08_apgar_distributions')

print(f'\nDone! {len(list(OUT.glob("*.png")))} PNGs saved to: {OUT.resolve()}')
