import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.reader(open('results.tsv'), delimiter='\t'))[1:]
xs, keep_x, keep_y, disc_x, disc_y = [], [], [], [], []
best, best_x, best_y = float('inf'), [], []
rope_x = None
for i, r in enumerate(rows, 1):
    val, status, desc = float(r[1]), r[3], r[4]
    valid = 0 < val < 1e9          # drop the crash (0) and the diverged run (3.5e11)
    if status == 'keep':
        if valid:
            best = min(best, val)
            keep_x.append(i); keep_y.append(val)
        if 'RoPE replaces' in desc:
            rope_x = i
    elif valid:
        disc_x.append(i); disc_y.append(val)
    best_x.append(i); best_y.append(best)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(disc_x, disc_y, s=22, c='#c9c9c9', label='discarded', zorder=2)
ax.scatter(keep_x, keep_y, s=34, c='#2e9e4f', label='kept', zorder=3)
ax.plot(best_x, best_y, c='#1f4fd8', lw=2.2, label='best so far', zorder=4)

ax.annotate(f'baseline\n{best_y[0]:,.0f}', (best_x[0], best_y[0]),
            textcoords='offset points', xytext=(8, 8), fontsize=9, color='#333')
ax.annotate(f'final\n{best_y[-1]:,.0f}', (best_x[-1], best_y[-1]),
            textcoords='offset points', xytext=(-6, 14), fontsize=9, color='#1f4fd8', ha='right')
if rope_x:
    ax.annotate('RoPE', (rope_x, best_y[rope_x-1]), textcoords='offset points',
                xytext=(6, -18), fontsize=9, color='#1f4fd8',
                arrowprops=dict(arrowstyle='->', color='#1f4fd8'))

ax.set_xlabel('experiment #')
ax.set_ylabel('validation loss (denoising score entropy)')
ax.set_title('Autoresearch: tuning TON-v1 over 52 experiments')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper right', frameon=False)
fig.tight_layout()
fig.savefig('ar_progress.png', dpi=120)
print(f'kept {len(keep_x)} shown, {len(disc_x)} discards shown; baseline {best_y[0]:,.0f} -> final {best_y[-1]:,.0f}')
