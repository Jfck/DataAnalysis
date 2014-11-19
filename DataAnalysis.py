import Tkinter, Tkconstants, tkFileDialog
import os
import math
import re

import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
from xlsxwriter.utility import xl_range

from filefilter import *



############## utilities ##################

def euler2matrix(euler_angles):
	'''convert a z-x-z euler angle(measured in degree) to a rotation matrix'''
	# convert to radians
	euler_rad = np.array(euler_angles)*math.pi/180.0
	a1,a2,a3 = euler_rad[:]
	c1,s1 = math.cos(a1),math.sin(a1)
	c2,s2 = math.cos(a2),math.sin(a2)
	c3,s3 = math.cos(a3),math.sin(a3)
	r0 = [c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1,  s1*s2]
	r1 = [c3*s1+c1*c2*s3,  c1*c2*c3-s1*s3, -c1*s2]
	r2 = [    s2*s3     ,       c3*s2    ,    c2 ] 
	return np.asarray([r0,r1,r2],np.float64)
	

def pointCloudfromPLY(filename):
	'''get point cloud from file, return a numpy array'''
	try:
		ply = open(filename,'r')
	except IOError,e:
		return None
	count = 0
	pointcloud = []
	for line in ply:
		count += 1
		if count > 12:
			line = line.strip().split()
			if len(line) > 3:
				pointcloud.append(line[0:3])
			else:
				break
	ply.close()
	return np.asarray(pointcloud[:-1],np.float64)

def avrgDist(data):
	square = data * data
	return np.sqrt(square.sum(1)).sum()/data.shape[0]

def get_avgdist_point_cloud(directory):
	files = FileFilt([".ply"])
	files.FindFile(directory)
	avg_dist = []
	for filename in files.fileList:
		try:
			pc = pointCloudfromPLY(filename)
			avg_dist.append(avrgDist(pc))
		except IOError,e:
			return None	
	return np.asarray(avg_dist,np.float64)

def get_single_analysis_data(directory):
	"""Return Experimental value & Theoretical value & Relative Error as numpy array"""
	rel_angle_err = []
	rel_trans_err = []
	exp_angle = []
	the_angle = []
	exp_trans = []
	the_trans = []
	exp_acc_angle = [[0.0,0.0,0.0],]
	the_acc_angle = [[0.0,0.0,0.0],]
	
	exp_acc_trans = []
	the_acc_trans = []	
	
	files = FileFilt([".txt"])
	files.FindFile(directory)
	
	file = open(files.fileList[0],'r')
	lines = file.readlines()
	line = lines[51]
	the_acc_trans.append(line.strip().split())
	line = lines[48]
	exp_acc_trans.append(line.strip().split())
	
	for filename in files.fileList:
		try:
			file = open(filename,'r')
		except IOError,e:
			return None,None
		
		lines = file.readlines()
		
		#	Get Relative Angle Error:
		line = lines[1]
		line = line.strip()
		rel_angle_err.append(line.split())			
		
		#	Get Relative Angle Error:
		line = lines[4]
		line = line.strip()
		rel_trans_err.append(line.split())		
		
		#	Get Experimental Euler Angles:
		line = lines[7]
		line = line.strip()
		exp_angle.append(line.split())	
		
		#	Get Theory Euler Angles:
		line = lines[10]
		line = line.strip()
		the_angle.append(line.split())	
		
		#	Get Experimental Translation:
		line = lines[13]
		line = line.strip()
		exp_trans.append(line.split())				

		#	Get Experimental Translation:
		line = lines[16]
		line = line.strip()
		the_trans.append(line.split())
		
		#	Get Accumulated Experimental Euler Angles:
		line = lines[30]
		line = line.strip()
		exp_acc_angle.append(line.split())	
		
		#	Get Accumulated Theory Euler Angles:
		line = lines[33]
		line = line.strip()
		the_acc_angle.append(line.split())
		
		#	Get Accumulated Experimental Translation:
		line = lines[36]
		line = line.strip()
		exp_acc_trans.append(line.split())	
		
		#	Get Accumulated Theory Translation:
		line = lines[39]
		line = line.strip()
		the_acc_trans.append(line.split())		
		
	return np.asarray(rel_angle_err,np.float64), np.asarray(rel_trans_err,np.float64),\
         np.asarray(exp_angle,np.float64)    , np.asarray(the_angle,np.float64)    ,\
         np.asarray(exp_trans,np.float64)    , np.asarray(the_trans,np.float64)    ,\
	       np.asarray(exp_acc_angle,np.float64), np.asarray(the_acc_angle,np.float64),\
	       np.asarray(exp_acc_trans,np.float64), np.asarray(the_acc_trans,np.float64)

def get_single_error_data(directory):
	"""Return the relative error data in the file as a numpy array"""
	rel_angle_err = []
	rel_trans_err = []
	
	files = FileFilt([".txt"])
	files.FindFile(directory)
	
	for filename in files.fileList:
		try:
			file = open(filename,'r')
		except IOError,e:
			return None,None
		
		lines = file.readlines()
		
		#	Get Relative Angle Error:
		line = lines[1]
		line = line.strip()
		rel_angle_err.append(line.split())
		
		#	Get Relative Angle Error:
		line = lines[4]
		line = line.strip()
		rel_trans_err.append(line.split())			
		
	return np.asarray(rel_angle_err,np.float64), np.asarray(rel_trans_err,np.float64)



########################################################################
class AnalysisApp(Tkinter.Frame):
	"""Analysis error of Stereo Measure"""

	#----------------------------------------------------------------------
	def __init__(self,root):
		"""Constructor"""
		Tkinter.Frame.__init__(self, root)
		
		# options for buttons
		button_opt = {'padx': 3, 'pady': 3}	
		
		# defining options for opening a directory
		self.dir_opt = options = {}
		options['initialdir'] = 'D:\\PyProjects'
		options['mustexist'] = False
		options['parent'] = root
		options['title'] = 'Choose a directory that contains data'		
		
		# file operation instructions
		self.mainframe = Tkinter.Frame(self)
		self.mainframe.grid(row = 0, column = 0)
		
		self.singlefiledirectory = Tkinter.StringVar()
		self.singlefiledirectory.set('Select the Single Data Directory')
		Tkinter.Label(self.mainframe,text = 'Single Data Path:').grid(row = 0, column = 0)
		Tkinter.Label(self.mainframe,textvariable = self.singlefiledirectory).grid(row = 0, column = 1)
		Tkinter.Button(self.mainframe, text='SDataDir', command=self.single_analysis).grid(row = 0, column = 2, **button_opt)
		
		self.wholefiledirectory = Tkinter.StringVar()
		self.wholefiledirectory.set('Select the Whole Data Directory')
		Tkinter.Label(self.mainframe,text = 'Whole Data Path:').grid(row = 1, column = 0)
		Tkinter.Label(self.mainframe,textvariable = self.wholefiledirectory).grid(row = 1, column = 1)
		Tkinter.Button(self.mainframe, text='WDataDir', command=self.whole_analysis).grid(row = 1, column = 2, **button_opt)	  	
		
	def single_analysis(self):
		'''analysis single floder data'''
		dirpath = tkFileDialog.askdirectory(**self.dir_opt)
		
		if not dirpath:
			return
		
		self.singlefiledirectory.set(dirpath)
		self.dir_opt['initialdir'] = dirpath
		
		#rel_angle_err,rel_trans_err = self.get_single_error_data(dirpath)
		rel_angle_err,rel_trans_err,\
		exp_angle,the_angle,\
		exp_trans,the_trans,\
		exp_acc_angle,the_acc_angle,\
		exp_acc_trans,the_acc_trans \
		   = get_single_analysis_data(dirpath+'//S_analysis')
#		avg_dist = get_avgdist_point_cloud(dirpath+'//S_pointCloud')
		
		filename = dirpath + '//' + 'Analysis.xlsx'
		self.write2xlsx_v0(filename,rel_angle_err,rel_trans_err,exp_angle,the_angle,exp_trans,the_trans)	
		
		#_,floder_name = os.path.split(dirpath)
		#ma = re.search(r'[0-9]{2}m', floder_name)
		#the_dist = np.asarray([ma.group()[0:2]]*avg_dist.shape[0],np.float64)
		filename = dirpath + '//' + 'Dist_Angle.xlsx'
		#self.write2xlsx_v1(filename, exp_acc_angle, the_acc_angle, avg_dist, the_dist)	
		
		self.write2xlsx_v2(filename, exp_acc_angle, the_acc_angle, exp_acc_trans, the_acc_trans)
		
		
		# plot threshold - average ralative error
		view_no = np.arange(0,exp_acc_angle.shape[0])
		
		f0 = plt.figure(figsize=(15,9))
		plt.title("Error Analysis")
	
		plt.subplot(2, 2, 1)
		plt.plot(view_no[1:-1],exp_acc_angle[1:-1,0],'r-',label="$Exp:Yaw$"  ,linewidth=1)
		plt.plot(view_no[1:-1],the_acc_angle[1:-1,0],'y-',label="$The:Yaw$"  ,linewidth=1)
		plt.ylabel("degree")			

		plt.legend(bbox_to_anchor=(0.65, 0.85, 0.34, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 2)
		plt.plot(view_no[1:-1],exp_acc_angle[1:-1,1],'g-',label="$Exp:Pitch$",linewidth=1)
		plt.plot(view_no[1:-1],the_acc_angle[1:-1,1],'y-',label="$The:Pitch$"  ,linewidth=1)
		plt.ylabel("degree")			

		plt.legend(bbox_to_anchor=(0.65, 0.85, 0.34, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 3)
		plt.plot(view_no[1:-1],exp_acc_angle[1:-1,2],'b-',label="$Exp:Roll$" ,linewidth=1)
		plt.plot(view_no[1:-1],the_acc_angle[1:-1,2],'y-',label="$The:Roll$"  ,linewidth=1)
		plt.xlabel("view_no")
		plt.ylabel("degree")			

		plt.legend(bbox_to_anchor=(0.65, 0.85, 0.34, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 4)
		#plt.plot(view_no,avg_dist,'c-',label="$Exp:Trans$",linewidth=1)
		#plt.plot(view_no,the_dist,'y-',label="$The:Trans$",linewidth=1)
		plt.plot(view_no[1:-1],exp_acc_trans[1:-1,0],'c-',label="$Exp:Trans$",linewidth=1)
		plt.plot(view_no[1:-1],exp_acc_trans[1:-1,1],'c-',linewidth=1)
		plt.plot(view_no[1:-1],exp_acc_trans[1:-1,2],'c-',linewidth=1)
		plt.plot(view_no[1:-1],the_acc_trans[1:-1,0],'y-',label="$The:Trans$",linewidth=1)
		plt.plot(view_no[1:-1],the_acc_trans[1:-1,1],'y-',linewidth=1)
		plt.plot(view_no[1:-1],the_acc_trans[1:-1,2],'y-',linewidth=1)						
		
		plt.xlabel("view_no")
		plt.ylabel("degree")		
	
		plt.legend(bbox_to_anchor=(0.65, 0.85, 0.34, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
	
		f0.show()		

		
		
		
		view_no = np.arange(0,rel_angle_err.shape[0])
		
		yaw = rel_angle_err[:,0]
		pitch = rel_angle_err[:,1]
		roll = rel_angle_err[:,2]
		
		# plot original data
		f1 = plt.figure(figsize=(15,9))
		plt.title("Error Analysis")
			
		plt.subplot(2, 2, 1)
		plt.plot(view_no,yaw*100,'ro-',label="$Yaw$"  ,linewidth=1)
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 30.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 2)
		plt.plot(view_no,pitch*100,'go-',label="$Pitch$",linewidth=1)
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 30.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 3)
		plt.plot(view_no,roll*100,'bo-',label="$Roll$" ,linewidth=1)
		plt.xlabel("view_no")
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 30.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 4)
		plt.plot(view_no,rel_trans_err * 100   ,'c.-',label="$Trans$",linewidth=1)
		plt.xlabel("view_no")
		plt.ylabel("percent(%)")		
		plt.ylim(0.0, 30.0)		
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		f1.show()	
		
		
		# vary threshold, compute average ralative error
		axis_threshold = np.arange(0.2, 1.05, 0.05)
		
		avg_angle_err = []
		avg_trans_err = []
		corr_ratio = []
		
		total_view_no = rel_trans_err.shape[0]	
		for threshold in axis_threshold:			
			ind = (rel_angle_err > threshold).sum(axis=1) + (rel_trans_err > threshold).reshape(total_view_no)
			ind = (ind < 1).nonzero()
			in_thres_view_no = ind[0].shape[0]
			in_thres_angle_err = rel_angle_err[ind]
			in_thres_trans_err = rel_trans_err[ind]
			
			single_avg_angle_err = in_thres_angle_err.sum(axis = 0)/in_thres_view_no
			single_avg_trans_err = in_thres_trans_err.sum()/in_thres_view_no
			
			avg_angle_err.append(single_avg_angle_err)
			avg_trans_err.append(single_avg_trans_err)
			corr_ratio.append(in_thres_view_no*100.0/total_view_no)		
		
		avg_angle_err = np.asarray(avg_angle_err,np.float64)
		avg_trans_err = np.asarray(avg_trans_err,np.float64)	
		axis_threshold = axis_threshold*100
		
		# plot threshold - average ralative error
		f2 = plt.figure(figsize=(15,9))
		plt.title("Error Analysis")
	
		plt.subplot(2, 2, 1)
		plt.plot(axis_threshold,avg_angle_err[:,0]*100,'ro-',label="$Yaw$"  ,linewidth=1)
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 30.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 2)
		plt.plot(axis_threshold,avg_angle_err[:,1]*100,'go-',label="$Pitch$",linewidth=1)
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 30.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 3)
		plt.plot(axis_threshold,avg_angle_err[:,2]*100,'bo-',label="$Roll$" ,linewidth=1)
		plt.xlabel("threshold(%)")
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 30.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 4)
		plt.plot(axis_threshold,avg_trans_err * 100   ,'c.-',label="$Trans$",linewidth=1)
		plt.xlabel("threshold(%)")
		plt.ylabel("percent(%)")		
		plt.ylim(0.0, 30.0)		
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		f2.show()
		
		# plot threshold - correct ratio
		f3 = plt.figure(figsize=(8,6))
		plt.title("Correct Ratio")
		plt.plot(axis_threshold,corr_ratio,'go-',label="$CorrentRatio$",linewidth=1)
		plt.xlabel("threshold(%)")
		plt.ylabel("percent(%)")		
		plt.ylim(0.0, 105.0)								
		f3.show()
		 
			
	def whole_analysis(self):
		"""analysis a set floders data"""
		dirpath = tkFileDialog.askdirectory(**self.dir_opt)
		if not dirpath:
			return
		
		self.wholefiledirectory.set(dirpath)
		self.dir_opt['initialdir'] = dirpath
		
		# get all single directories
		fileList = []	
		for s in os.listdir(dirpath):
			newDir = os.path.join(dirpath,s)
			if os.path.isdir(newDir):	
				newDir = os.path.join(newDir,'S_analysis')
				if os.path.exists(newDir):
					fileList.append(newDir)
					
		avg_angle_err = []
		avg_trans_err = []
		corr_ratio = []
		threshold = 0.10
		for singledir in fileList:
			rel_angle_err,rel_trans_err = get_single_error_data(singledir)
			total_view_no = rel_trans_err.shape[0]
			ind = (rel_angle_err > threshold).sum(axis=1) + (rel_trans_err > threshold).reshape(total_view_no)
			ind = (ind < 1).nonzero()
			in_thres_view_no = ind[0].shape[0]
			in_thres_angle_err = rel_angle_err[ind]
			in_thres_trans_err = rel_trans_err[ind]
			
			single_avg_angle_err = in_thres_angle_err.sum(axis = 0)/in_thres_view_no
			single_avg_trans_err = in_thres_trans_err.sum()/in_thres_view_no
			
			avg_angle_err.append(single_avg_angle_err)
			avg_trans_err.append(single_avg_trans_err)
			corr_ratio.append(in_thres_view_no*100.0/total_view_no)
		
		avg_angle_err = np.asarray(avg_angle_err,np.float64)
		avg_trans_err = np.asarray(avg_trans_err,np.float64)
		
		# for baseline
		
		#axis_baseline = np.arange(0.1, avg_angle_err.shape[0]*0.1+0.1, 0.1)
		
		#f1 = plt.figure(figsize=(15,9))
		#plt.title("Error Analysis")
			
		#plt.subplot(2, 2, 1)
		#plt.plot(axis_baseline,avg_angle_err[:,0]*100,'ro-',label="$Yaw$"  ,linewidth=1)
		#plt.ylabel("percent(%)")			
		#plt.ylim(0.0, 30.0)
		#plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		#plt.subplot(2, 2, 2)
		#plt.plot(axis_baseline,avg_angle_err[:,1]*100,'go-',label="$Pitch$",linewidth=1)
		#plt.ylabel("percent(%)")			
		#plt.ylim(0.0, 30.0)
		#plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		#plt.subplot(2, 2, 3)
		#plt.plot(axis_baseline,avg_angle_err[:,2]*100,'bo-',label="$Roll$" ,linewidth=1)
		#plt.xlabel("baseline(m)")
		#plt.ylabel("percent(%)")			
		#plt.ylim(0.0, 30.0)
		#plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		#plt.subplot(2, 2, 4)
		#plt.plot(axis_baseline,avg_trans_err * 100   ,'c.-',label="$Trans$",linewidth=1)
		#plt.xlabel("baseline(m)")
		#plt.ylabel("percent(%)")
		#plt.ylim(0.0, 30.0)		
		#plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		#f1.show()
		
		#f2 = plt.figure(figsize=(5,3))
		#plt.title("Correct Ratio")
		#plt.plot(axis_baseline,corr_ratio,'go-',label="$CorrentRatio$",linewidth=1)
		#plt.xlabel("baseline(m)")
		#plt.ylabel("percent(%)")
		#plt.ylim(0.0, 100.0)
		
		#f2.show()		
		
		#for FOV
		
		axis_FOV = np.arange(16, avg_angle_err.shape[0]*2+16, 2)
		
		f1 = plt.figure(figsize=(15,9))
		plt.title("Error Analysis")
			
		plt.subplot(2, 2, 1)
		plt.plot(axis_FOV,avg_angle_err[:,0]*100,'ro-',label="$Yaw$"  ,linewidth=1)
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 15.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 2)
		plt.plot(axis_FOV,avg_angle_err[:,1]*100,'go-',label="$Pitch$",linewidth=1)
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 15.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 3)
		plt.plot(axis_FOV,avg_angle_err[:,2]*100,'bo-',label="$Roll$" ,linewidth=1)
		plt.xlabel("FOV(degree)")
		plt.ylabel("percent(%)")			
		plt.ylim(0.0, 15.0)
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		plt.subplot(2, 2, 4)
		plt.plot(axis_FOV,avg_trans_err * 100   ,'c.-',label="$Trans$",linewidth=1)
		plt.xlabel("FOV(degree)")
		plt.ylabel("percent(%)")		
		plt.ylim(0.0, 15.0)		
		plt.legend(bbox_to_anchor=(0.75, 0.85, 0.24, 0.1), loc=1,ncol=1, mode="expand", borderaxespad=0.)
		
		f1.show()
		
		
		f2 = plt.figure(figsize=(5,3))
		plt.title("Correct Ratio")
		plt.plot(axis_FOV,corr_ratio,'go-',label="$CorrentRatio$",linewidth=1)
		plt.xlabel("FOV(degree)")
		plt.ylabel("percent(%)")		
		plt.ylim(0.0, 100.0)						
		
		f2.show()

	def write2xlsx_v0(self, fliename, rel_angle_err, rel_trans_err, exp_angle, the_angle, exp_trans, the_trans):
		
		# Create an new Excel file and add a worksheet.
		workbook = xlsxwriter.Workbook(fliename)
		worksheet = workbook.add_worksheet()		
		
		# format 
		title_format = workbook.add_format({'align': 'center',
			                                 'valign': 'vcenter',
			                                 'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1})
		ev_format = workbook.add_format({'align': 'center',
			                                'valign': 'vcenter',
			                                'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                                'bg_color': '#F0F0F0',
			                               'num_format': '0.000'}) 	
		tv_format = workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEEED1',
			                               'num_format': '0.000'})                                 	
		re_format = workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEE8CD',
			                               'num_format': 10,
			                               'bold' : 1})
		highlight_re_format =  workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEE8CD', 
			                               'font_color': '#FF0000',
			                               'num_format': 10,
			                               'bold' : 1})
		# write first row
		worksheet.write('A1','No',title_format)
		worksheet.write('B1','Type',title_format)
		worksheet.merge_range('C1:E1','Euler Angle (degree)',title_format)
		worksheet.merge_range('F1:H1','Translation (m)',title_format)	
		
		for i in range(0,rel_angle_err.shape[0]):
			first_row = i*3+1
			last_row = i*3+3
			# write the number
			r = xl_range(first_row,0,last_row,0)
			worksheet.merge_range(r,i+1,title_format)
			
			# write Experimental value
			cur_row = first_row
			worksheet.write(cur_row, 1,'Experimental value',ev_format)
			worksheet.write_row(cur_row, 2, exp_angle[i], ev_format)
			worksheet.write_row(cur_row, 5, exp_trans[i], ev_format)		
			
			# write Theoretical value
			cur_row = cur_row + 1
			worksheet.write(cur_row, 1,'Theoretical value',tv_format)
			worksheet.write_row(cur_row, 2, the_angle[i], tv_format)
			worksheet.write_row(cur_row, 5, the_trans[i], tv_format)
			
			# write Relative Error
			cur_row = cur_row + 1
			threshold = 0.3
			worksheet.write(cur_row, 1,'Relative Error',re_format)
			angle_err = rel_angle_err[i]
			r = xl_range(cur_row,5,cur_row,7)
			if (angle_err > threshold).sum() or (rel_trans_err[i] > threshold).sum():
				worksheet.write_row(cur_row, 2, angle_err, highlight_re_format)
				worksheet.merge_range(r,rel_trans_err[i],highlight_re_format)
			else:	
				worksheet.write_row(cur_row, 2, angle_err, re_format)
				worksheet.merge_range(r,rel_trans_err[i],re_format)			
		
		# set col weight
		worksheet.set_column(0,0,2.5)
		worksheet.set_column(1,1,18)
		worksheet.set_column(2,7,6.8)
		workbook.close()

	def write2xlsx_v1(self, fliename, exp_acc_angle, the_acc_angle, exp_dist, the_dist):
		
		# Create an new Excel file and add a worksheet.
		workbook = xlsxwriter.Workbook(fliename)
		worksheet = workbook.add_worksheet()		
		
		# format 
		title_format = workbook.add_format({'align': 'center',
			                                 'valign': 'vcenter',
			                                 'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1})
		ev_format = workbook.add_format({'align': 'center',
			                                'valign': 'vcenter',
			                                'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                                'bg_color': '#F0F0F0',
			                               'num_format': '0.000'}) 	
		tv_format = workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEEED1',
			                               'num_format': '0.000'})                                 	
		re_format = workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEE8CD',
			                               'num_format': 10,
			                               'bold' : 1})
		highlight_re_format =  workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEE8CD', 
			                               'font_color': '#FF0000',
			                               'num_format': 10,
			                               'bold' : 1})
		# write first row
		worksheet.write('A1','No',title_format)
		worksheet.write('B1','Type',title_format)
		worksheet.merge_range('C1:E1','Accumulated Euler Angle(degree)',title_format)
		worksheet.write('F1','Avrg Dist(m)',title_format)	
		
		for i in range(0,exp_acc_angle.shape[0]):
			first_row = i*3+1
			last_row = i*3+3
			# write the number
			r = xl_range(first_row,0,last_row,0)
			worksheet.merge_range(r,i,title_format)
			
			# write Experimental value
			cur_row = first_row
			worksheet.write(cur_row, 1,'Experimental value',ev_format)
			worksheet.write_row(cur_row, 2, exp_acc_angle[i], ev_format)
			worksheet.write(cur_row, 5, exp_dist[i], ev_format)		
			
			# write Theoretical value
			cur_row = cur_row + 1
			worksheet.write(cur_row, 1,'Theoretical value',tv_format)
			worksheet.write_row(cur_row, 2, the_acc_angle[i], tv_format)
			worksheet.write(cur_row, 5, the_dist[i], tv_format)
			
			# write Relative Error
			cur_row = cur_row + 1
			worksheet.write(cur_row, 1,'Relative Error',re_format)
			r = xl_range(cur_row,2,cur_row,5)
			r1 = xl_range(first_row,2,first_row,5)
			r2 = xl_range(first_row+1,2,first_row+1,5)
			worksheet.write_array_formula(r,'ABS(('+r2+'-'+r1+')/'+r2+')',cell_format = re_format)
					
		# set col weight
		worksheet.set_column(0,0,2.5)
		worksheet.set_column(1,1,18)
		worksheet.set_column(2,4,9.5)
		worksheet.set_column(5,5,12)
		workbook.close()


	def write2xlsx_v2(self, fliename, exp_acc_angle, the_acc_angle, exp_acc_trans, the_acc_trans):
		the_init_position = the_acc_trans[0]
		exp_init_position = exp_acc_trans[0]
		
		init_err = np.sqrt(((the_init_position-exp_init_position)**2).sum())/np.sqrt((the_init_position*2).sum())
		
		the_other_position = the_acc_trans[1:]
		exp_other_position = exp_acc_trans[1:]
		
		err_other_position = np.abs(the_other_position - exp_other_position)
		err_other_position = err_other_position ** 2
		err_other_position = np.sqrt(err_other_position.sum(axis=1))/np.sqrt((the_init_position**2).sum())
		
		# Create an new Excel file and add a worksheet.
		workbook = xlsxwriter.Workbook(fliename)
		worksheet = workbook.add_worksheet()		
		
		# format 
		title_format = workbook.add_format({'align': 'center',
			                                 'valign': 'vcenter',
			                                 'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1})
		ev_format = workbook.add_format({'align': 'center',
			                                'valign': 'vcenter',
			                                'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                                'bg_color': '#F0F0F0',
			                               'num_format': '0.000'}) 	
		tv_format = workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEEED1',
			                               'num_format': '0.000'})                                 	
		re_format = workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEE8CD',
			                               'num_format': 10,
			                               'bold' : 1})
		highlight_re_format =  workbook.add_format({'align': 'center',
			                               'valign': 'vcenter',
			                               'border': 1, 'bottom': 1, 'top': 1, 'left': 1, 'right': 1,
			                               'bg_color': '#EEE8CD', 
			                               'font_color': '#FF0000',
			                               'num_format': 10,
			                               'bold' : 1})
		# write first row
		worksheet.write('A1','No',title_format)
		worksheet.write('B1','Type',title_format)
		worksheet.merge_range('C1:E1','Acc Euler Angle(degree)',title_format)
		worksheet.merge_range('F1:H1','Acc Translation(m)',title_format)
		
		r = xl_range(1,0,3,0)
		worksheet.merge_range(r,0,title_format)
		worksheet.write(1, 1,'Exp init value',ev_format)
		worksheet.write_row(1, 2, exp_acc_angle[0], ev_format)
		worksheet.write_row(1, 5, exp_acc_trans[0], ev_format)
		
		worksheet.write(2, 1,'The init value',tv_format)
		worksheet.write_row(2, 2, the_acc_angle[0], tv_format)
		worksheet.write_row(2, 5, the_acc_trans[0], tv_format)
		
		worksheet.write(3, 1,'Relative Error',re_format)
		worksheet.write_row(3, 2, [0,0,0], re_format)
		r = xl_range(3,5,3,7)
		worksheet.merge_range(r,init_err, re_format)		
		
		for i in range(1,exp_acc_angle.shape[0]):
			first_row = i*3+1
			last_row = i*3+3
			# write the number
			r = xl_range(first_row,0,last_row,0)
			worksheet.merge_range(r,i,title_format)
			
			# write Experimental value
			cur_row = first_row
			worksheet.write(cur_row, 1,'Experimental value',ev_format)
			worksheet.write_row(cur_row, 2, exp_acc_angle[i], ev_format)
			worksheet.write_row(cur_row, 5, exp_acc_trans[i], ev_format)		
			
			# write Theoretical value
			cur_row = cur_row + 1
			worksheet.write(cur_row, 1,'Theoretical value',tv_format)
			worksheet.write_row(cur_row, 2, the_acc_angle[i], tv_format)
			worksheet.write_row(cur_row, 5, the_acc_trans[i], tv_format)
			
			# write Relative Error
			cur_row = cur_row + 1
			worksheet.write(cur_row, 1,'Relative Error',re_format)
			r = xl_range(cur_row,2,cur_row,4)
			r1 = xl_range(first_row,2,first_row,4)
			r2 = xl_range(first_row+1,2,first_row+1,4)
			worksheet.write_array_formula(r,'ABS(('+r2+'-'+r1+')/'+r2+')',cell_format = re_format)
			r = xl_range(cur_row,5,cur_row,7)			
			worksheet.merge_range(r,err_other_position[i-1],re_format)
					
		# set col weight
		worksheet.set_column(0,0,2.5)
		worksheet.set_column(1,1,18)
		worksheet.set_column(2,7,6.8)
		workbook.close()
		
if __name__=='__main__':
	
	root = Tkinter.Tk()
	AnalysisApp(root).pack()
	root.mainloop()