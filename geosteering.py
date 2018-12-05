import random
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

r = random.Random()
r.seed(42)

class geosteering:

	def dip_offset(self,dip,x):
		'''
		'''

		return x * math.tan(math.radians(dip))


	def drill_step(self,m,inc,comp,d,x,i):
		'''
		'''

		if m < self.md_bend:
			d = d - i
			x = x
			inc = 0
		else:
			d = d - (i * math.cos(math.radians(inc)))
			x = x + (i * math.sin(math.radians(inc)))

			if inc < 90:
			 	inc = min(inc + (self.inc_build_max * r.random()),self.inc_max)
			else:
				inc = inc + (max(-self.inc_delta_max,min(self.inc_delta_max,(comp))))
				#inc = inc + comp

			# if inc < 90:
			# 	inc = min(inc + (self.inc_build_max * r.random()),self.inc_max)
			# else:
			# 	inc = min(inc + (self.inc_build_max * r.random()),self.inc_max) + max(-self.inc_delta_max,min(self.inc_delta_max,(self.dampening * self.wingrad[1])))
			# inc = min(inc + (self.inc_build_max * r.random()), self.inc_max) + (self.dampening * next_turn * log_dwin)

		return inc,d,x

	def get_log(self,tvd,x):
		'''
		'''

		try:
			strip = self.slice[self.slice['x'] == int(x)]
			idx = abs(strip['tvd'] - tvd).idxmin()
			log = strip['log'][idx]
			zone = strip['zone'][idx]
		except:
			log = pd.np.nan
			zone = pd.np.nan
		return [zone,log]

	def accumulator(self,window,val):
		'''
		'''

		try:
			d = self.well[val][-1] - self.well[val][-2]
		except:
			d = 0

		try:
			dwin = np.mean(self.well[val+'_d'][-window:])
		except:
			dwin = 0

		return [d, dwin]

	def generate_strat(self):
		'''
		'''
		
		self.log = []
		self.zone = []
		self.tvd = []
		td = 0
		for z in range(self.zone_ct):
			zone_thickness = r.randint(self.zone_ht_min,self.zone_ht_max)
			val = 0
			for d in range(zone_thickness):
				self.tvd.append(td)
				self.zone.append(z)
				self.log.append(val)
				if d < zone_thickness*(1/5):
					val = max(0,val + r.uniform(-1,2))
				else:
					val = max(0,val - r.uniform(-1,1))
				td += 1

		self.td = td
		self.tvd = list(-np.array(self.tvd))

		self.strat = pd.DataFrame()
		self.strat['tvd'] = self.tvd
		self.strat['zone'] = self.zone
		self.strat['log'] = self.log

	def generate_slice(self):
		'''
		'''

		self.slice = pd.DataFrame()
		offset = 0
		for x in range(self.surface_dist):
			col = pd.DataFrame()
			col['tvd'] = self.tvd
			col['tvd'] = col['tvd'] + offset
			col['x'] = x
			col['zone'] = self.zone
			col['log'] = self.log

			self.slice = self.slice.append(col)

			dip = self.dip_offset(r.randint(self.dip_min,self.dip_max),1)
			if r.random() < self.fault_prob:
				fault = r.randint(0,self.fault_shift_max)
			else:
				fault = 0
			offset = offset + dip - fault

		self.slice_upscale = self.slice.copy(deep=True)
		self.slice_upscale['tvd'] = self.slice_upscale['tvd'].round()
		self.slice_im_log = pd.pivot_table(self.slice_upscale, values='log', index=['tvd'], columns=['x'], aggfunc=np.mean)
		self.slice_im_zone = pd.pivot_table(self.slice_upscale, values='zone', index=['tvd'], columns=['x'], aggfunc=np.mean)

	def drill_well(self):
		'''
		'''

		x = 0
		inc = 0
		tvd = 0
		next_turn = 0

		self.well = {}
		self.well['md'] = []
		self.well['tvd'] = []
		self.well['x'] = []
		self.well['inc'] = []
		self.well['inc_d'] = []
		self.well['inc_dwin'] = []
		self.well['next_turn'] = []
		self.well['log'] = []
		self.well['zone'] = []
		self.well['log_d'] = []
		self.well['log_dwin'] = []
		self.well['comp'] = []

		for m in range(self.md_max):
			self.well['md'].append(m)
			self.well['tvd'].append(tvd)
			self.well['x'].append(x)
			self.well['inc'].append(inc)
			acc_inc = self.accumulator(self.window, 'inc')
			self.well['inc_d'].append(acc_inc[0])
			self.well['inc_dwin'].append(acc_inc[1])
			self.well['next_turn'].append(next_turn)
			zonelog = self.get_log(tvd,x)
			self.well['zone'].append(zonelog[0])
			self.well['log'].append(zonelog[1])
			acc_log = self.accumulator(self.window, 'log')
			self.well['log_d'].append(acc_log[0])
			self.well['log_dwin'].append(acc_log[1])
			comp = self.dampening * next_turn * min(0,acc_log[1])
			self.well['comp'].append(comp)
			inc,tvd,x = self.drill_step(m,inc,comp,tvd,x,1)
			if acc_inc[1] < 0:
				next_turn = -1
			elif acc_inc[1] > 0:
				next_turn = 1

		self.well = pd.DataFrame(self.well)

	def plot_slice(self):
		'''
		'''
		
		ax1 = plt.subplot(211)
		#plt.scatter(self.slice['x'],self.slice['tvd'],c=self.slice['zone'])
		plt.imshow(self.slice_im_zone, extent=[0,self.surface_dist,self.slice_im_zone.index.min(),self.slice_im_zone.index.max()], origin='lower')
		plt.plot(self.well['x'], self.well['tvd'], '-k')
		plt.setp(ax1.get_xticklabels(), visible=False)
		plt.title('x-z slice')
		plt.xlim(0,self.surface_dist)
		plt.ylim(self.slice_im_zone.index.min(),self.slice_im_zone.index.max())
		plt.ylabel('tvd')
		#plt.axis('equal')

		plt.subplot(212, sharex=ax1)
		#plt.scatter(self.slice['x'],self.slice['tvd'],c=self.slice['log'])
		plt.imshow(self.slice_im_log, extent=[0,self.surface_dist,self.slice_im_zone.index.min(),self.slice_im_zone.index.max()], origin='lower')
		plt.plot(self.well['x'], self.well['tvd'], '-k')
		plt.xlim(0,self.surface_dist)
		plt.ylim(self.slice_im_log.index.min(),self.slice_im_log.index.max())
		plt.ylabel('tvd')
		plt.xlabel('x')

		plt.savefig('xz_slice.png', dpi=600)
		plt.gcf().clear()

	def plot_strat(self):
		'''
		'''

		ax1 = plt.subplot(121)
		plt.scatter(self.strat['zone'],self.strat['tvd'],c=self.strat['zone'])
		plt.title('zone')
		plt.xlim(self.strat['tvd'].min(),self.strat['tvd'].max())
		plt.ylabel('tvd')
		plt.xlabel('zone')
		plt.axis('tight')

		ax2 = plt.subplot(122, sharey=ax1)
		plt.scatter(self.strat['log'],self.strat['tvd'],c=self.strat['log'])
		plt.setp(ax2.get_yticklabels(), visible=False)
		plt.title('log')
		plt.xlim(self.strat['tvd'].min(),self.strat['tvd'].max())
		plt.xlabel('log')

		plt.savefig('strat.png', dpi=600)
		plt.gcf().clear()

	def plot_well(self):
		'''
		'''
		
		ax1 = plt.subplot(211)
		plt.imshow(self.slice_im_zone, extent=[0,self.surface_dist,self.slice_im_zone.index.min(),self.slice_im_zone.index.max()], origin='lower', alpha=0.4)
		plt.scatter(self.well['x'], self.well['tvd'], c=self.well['zone'], s=1)
		plt.setp(ax1.get_xticklabels(), visible=False)
		plt.title('well log')
		plt.xlim(0,self.surface_dist)
		plt.ylim(self.slice_im_zone.index.min(),self.slice_im_zone.index.max())
		plt.ylabel('tvd')
		#plt.axis('equal')

		plt.subplot(212, sharex=ax1)
		plt.imshow(self.slice_im_log, extent=[0,self.surface_dist,self.slice_im_log.index.min(),self.slice_im_log.index.max()], origin='lower', alpha=0.4)
		plt.scatter(self.well['x'], self.well['tvd'], c=self.well['log'], s=1)
		plt.setp(ax1.get_xticklabels(), visible=False)
		plt.xlim(0,self.surface_dist)
		plt.ylim(self.slice_im_log.index.min(),self.slice_im_log.index.max())
		plt.ylabel('tvd')
		plt.xlabel('x')

		plt.savefig('well_log.png', dpi=600)
		plt.gcf().clear()

	def plot_wellqc(self):
		'''
		'''
		
		ax1 = plt.subplot(211)
		plt.imshow(self.slice_im_log, extent=[0,self.surface_dist,self.slice_im_log.index.min(),self.slice_im_log.index.max()], origin='lower', alpha=0.4)
		plt.scatter(self.well['x'], self.well['tvd'], c=self.well['log_d'], s=1)
		plt.setp(ax1.get_xticklabels(), visible=False)
		plt.xlim(0,self.surface_dist)
		plt.ylim(self.slice_im_log.index.min(),self.slice_im_log.index.max())
		plt.ylabel('tvd')

		plt.subplot(212, sharex=ax1)
		plt.imshow(self.slice_im_log, extent=[0,self.surface_dist,self.slice_im_log.index.min(),self.slice_im_log.index.max()], origin='lower', alpha=0.4)
		plt.scatter(self.well['x'], self.well['tvd'], c=self.well['log_dwin'], s=1)
		plt.xlim(0,self.surface_dist)
		plt.ylim(self.slice_im_log.index.min(),self.slice_im_log.index.max())
		plt.ylabel('tvd')
		plt.xlabel('x')

		plt.savefig('well_logqc.png', dpi=600)
		plt.gcf().clear()

	def __init__(self):
		'''
		'''

		self.zone_ct = 8
		self.zone_ht_min = 5
		self.zone_ht_max = 100
		self.generate_strat()

		self.surface_dist = 1500
		self.dip_min = 4
		self.dip_max = 10
		self.fault_prob = 0.02
		self.fault_shift_max = 10
		self.generate_slice()

		self.md_bend = 100
		self.md_lat = 450
		self.inc_min = 60
		self.inc_max = 120
		self.inc_build_max = 0.6
		self.md_max = 1500
		self.window = 5
		self.dampening = 20
		self.inc_delta_max = 5
		self.drill_well()

		self.plot_slice()
		self.plot_strat()
		self.plot_well()
		self.plot_wellqc()

a = geosteering()

b = a.well
plt.scatter(b['log'],b['log_dwin'])
