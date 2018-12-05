import matplotlib.pyplot as plt
import numpy as np

def vogel(pwf,pr,qo,plot):
	qomax = qo / (1 - 0.2*(pwf/pr) - 0.8*(pwf/pr)**2)
	pwf = 0
	pwf_l = []
	qo_l = []
	while pwf <= pr:
		pwf_l.append(pwf)
		qo_l.append(qomax * (1 - 0.2*(pwf/pr) - 0.8*(pwf/pr)**2))
		pwf += 1

	if plot:
		plt.plot(qo_l,pwf_l)
		plt.xlabel('q')
		plt.ylabel('pwf')
		plt.title('vogel ipr')
		plt.show()

	return pwf_l, qo_l, qomax

def fetkovich(pwf,pr,qo,plot):
	n = np.polyfit(np.log(pwf), np.log(qo), 1)
	qomax = qo / (1 - (pwf/pr)**2)**n
	pwf = 0
	pwf_l = []
	qo_l = []
	while pwf <= pr:
		pwf_l.append(pwf)
		qo_l.append(qomax * (1 - 0.2*(pwf/pr) - 0.8*(pwf/pr)**2))
		pwf += 1

	if plot:
		plt.plot(qo_l,pwf_l)
		plt.xlabel('q')
		plt.ylabel('pwf')
		plt.title('vogel ipr')
		plt.show()

	return pwf_l, qo_l, qomax

pwf_l, qo_l, qomax = vogel(1900,2400,400,True)

#def prod_index():