import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.reader(open('results_v2.tsv'), delimiter='\t'))[1:]
xs, keep_x, keep_y, disc_x, disc_y = [], [], [], [], []
best, best_x, best_y = float('inf'), [], []
mdlm_x = None
for i, r in enumerate(rows, 1):
    ppl, status, desc = float(r[2]), r[5], r[6]
    if status == 'keep':
        best = min(best, ppl)
        keep_x.append(i); keep_y.append(ppl)
        if 'MDLM masked/absorbing' in desc:
            mdlm_x = i
    else:
        disc_x.append(i); disc_y.append(ppl)
    best_x.append(i); best_y.append(best)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(disc_x, disc_y, s=22, c='#c9c9c9', label='discarded', zorder=2)
ax.scatter(keep_x, keep_y, s=34, c='#2e9e4f', label='kept', zorder=3)
ax.plot(best_x, best_y, c='#1f4fd8', lw=2.2, label='best so far', zorder=4)

ax.annotate(f'SEDD baseline\n{best_y[0]:,.0f}', (best_x[0], best_y[0]),
            textcoords='offset points', xytext=(20, -18), fontsize=9, color='#333')
ax.annotate(f'final\n{best_y[-1]:,.1f}', (best_x[-1], best_y[-1]),
            textcoords='offset points', xytext=(-6, 14), fontsize=9, color='#1f4fd8', ha='right')
if mdlm_x:
    ax.annotate('switch to MDLM', (mdlm_x, best_y[mdlm_x - 1]), textcoords='offset points',
                xytext=(10, 30), fontsize=9, color='#1f4fd8',
                arrowprops=dict(arrowstyle='->', color='#1f4fd8'))

ax.set_xlabel('experiment #')
ax.set_ylabel('generative perplexity (GPT-2, lower is better)')
ax.set_title('Autoresearch: architecture + loss search guided by gen_ppl (autoresearch_v2)', pad=14)
ax.grid(True, alpha=0.25)
ax.legend(loc='upper right', frameon=False)
fig.tight_layout()
fig.savefig('ar_progress_mdlm.png', dpi=120)
print(f'kept {len(keep_x)}, discarded {len(disc_x)}; baseline {best_y[0]:,.1f} -> final {best_y[-1]:,.1f}')
