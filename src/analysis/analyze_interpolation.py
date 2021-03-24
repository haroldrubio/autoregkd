import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass, fields

# Interpolation probabilities
probs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# Metrics
baseline_em = 86.1
baseline_f1 = 89.2

f1 = [84.94, 85.08, 83.369, 83.283, 81.237, 76.115, 74.35]
em = [81.588, 81.681, 79.693, 79.794, 77.419, 71.802, 69.856]
f1_retained = f1[1]/baseline_f1
em_retained = em[2]/baseline_em
bart_f1 = baseline_f1 * np.ones(len(f1))
bart_em = baseline_em * np.ones(len(em))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), sharex=True)
fnt_size = 16
	
ax1.plot(probs, f1, 'bo-', label='Interpolation SQuAD v2 F1')
ax2.plot(probs, em, 'ro-', label='Interpolation SQuAD v2 EM')
ax1.plot(probs, bart_f1, 'k--')
ax2.plot(probs, bart_em, 'k--')
fig.text(0.5, 0.04, 'Swapping Probability', ha='center', va='center', fontsize=fnt_size)
fig.text(0.33, 0.85, 'BART-large Baseline', fontsize=fnt_size)
fig.text(0.75, 0.85, 'BART-large Baseline', fontsize=fnt_size)
fig.text(0.15, 0.75, f'{f1_retained*100:.2f}% F1 retained', fontsize=fnt_size)
fig.text(0.57, 0.75, f'{em_retained*100:.2f}% EM retained', fontsize=fnt_size)

ax1.set(ylabel='F1')
ax2.set(ylabel='EM')
ax1.yaxis.label.set_size(fnt_size)
ax2.yaxis.label.set_size(fnt_size)
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(fnt_size)
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
	label.set_fontsize(fnt_size)

plt.suptitle('SQuAD v2 Dev Set Performance on Interpolation Networks', fontsize=fnt_size+2)
leg1 = ax1.legend(fontsize=fnt_size)
leg2 = ax2.legend(fontsize=fnt_size)
plt.show()

'''
@dataclass
class testData():
	a: int = 3

c = testData(a=2)
for f in fields(c):
	print(f.name)
	print(getattr(c, f.name))
print(c.a)
'''