import random
import numpy as np
from scipy.stats import loguniform

SIZE = 500
OUT_FN = 'hypset_script.sh'

np.random.seed()
rand_lrs = loguniform.rvs(10**-8, 0.05, size=SIZE)
rand_lambdas = loguniform.rvs(10**-5, 0.1, size=SIZE)
rand_grad_acc_steps = random.choices([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], k=SIZE)

rand_lrs_str = ' '.join(['%.12f' % x for x in rand_lrs])
rand_lrs_str = f"lrs=({rand_lrs_str})\n"

rand_lambdas_str = ' '.join(['%.10f' % x for x in rand_lambdas])
rand_lambdas_str = f"lambdas=({rand_lambdas_str})\n"

rand_grad_acc_steps_str = ' '.join(['%d' % x for x in rand_grad_acc_steps])
rand_grad_acc_steps_str = f"steps=({rand_grad_acc_steps_str})\n"

with open(OUT_FN, 'w', encoding='utf8') as ofp:
	ofp.write(rand_lrs_str)
	ofp.write(rand_lambdas_str)
	ofp.write(rand_grad_acc_steps_str)

