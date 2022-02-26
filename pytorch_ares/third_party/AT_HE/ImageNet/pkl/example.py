def countjson():
	import pickle
	names = ['pgd','pgdHE']
	for name in names:
		epoch = 70
		with open("{}_1000_t_epoch{}.pkl".format(name, epoch), "rb") as f:
			weight_clean = pickle.load(f)
		with open("{}_1000_adv_t_epoch{}.pkl".format(name, epoch), "rb") as g:
			weight_adv = pickle.load(g)	
		for key in weight_clean.keys():
			print("{}\t{} clean, mean:{} std:{}".format(name, key, weight_clean[key].mean(), weight_clean[key].std()))
			print("{}\t{} adv, mean:{} std:{}".format(name, key, weight_adv[key].mean(), weight_adv[key].std()))



def print_for_matlab():
	import pickle
	#names = ['pgd','pgdHE']
	names = ['trades','tradesHE']
	epoch = 70
	for name in names:
		key_names = []
		mean_cle = []
		mean_adv = []
		with open("{}_1000_t_epoch{}.pkl".format(name, epoch), "rb") as f:
			weight_clean = pickle.load(f)
		with open("{}_1000_adv_t_epoch{}.pkl".format(name, epoch), "rb") as g:
			weight_adv = pickle.load(g)
		print('Print for '+ name)	
		for key in weight_clean.keys():
			key_names.append(key)
			mean_cle.append(round(weight_clean[key].mean(),5))
			mean_adv.append(round(weight_adv[key].mean(),5))
		print('key names: ', key_names)
		print('clean means: ', mean_cle)
		print('adv means: ', mean_adv)