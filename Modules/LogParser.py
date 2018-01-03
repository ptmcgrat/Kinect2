import os, cv2, seaborn
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from roipoly import roipoly
import pylab as pl

class LogParser:    
    def __init__(self, logfile):
        self.logfile = logfile
        self.master_directory = logfile.replace(logfile.split('/')[-1], '')
        self.depth_df = self.master_directory + 'Depth.npy'

    def parse_log(self):

        self.npy_files = []
        self.jpg_files = []
        self.video_files = []
        self.npy_data = [] #stored as tuple (n_frames, med,min,max,std,gp)
        
        for line in self.lf:
            info_type = line.split(':')[0]
            if info_type == 'FrameCaptured':
                npy_file = line.split(': ')[1].split(',')[0]
                jpg_file = npy_file.replace('.npy', '.jpg')
                fnframe = int(line.split('NFrames: ')[1].split(',')[0])
                fmed = float(line.split('Med: ')[1].split(',')[0])
                fmin = float(line.split('Min: ')[1].split(',')[0])
                fmax = float(line.split('Max: ')[1].split(',')[0])
                fstd = float(line.split('Std: ')[1].split(',')[0])
                fpix = int(line.split('GP: ')[1].split(' of')[0])

                self.npy_files.append(self.master_directory + npy_file)
                self.jpg_files.append(self.master_directory + jpg_file)
                self.npy_data.append((fnframe, fmed,fmin,fmax,fstd,fpix))
                
            if info_type == 'ROI':
                self.background_image = cv2.imread(self.master_directory + line.rstrip().split('Image: ')[1].split(',')[0])
                print(line.rstrip().split('Shape: ('))
                self.width = int(line.rstrip().split('Shape: (')[1].split(',')[3].split(')')[0])
                self.height = int(line.rstrip().split('Shape: (')[1].split(',')[2])

            if info_type == 'PiCameraStarted':
                video_file = line.rstrip().split('File=')[1].split(',')[0]
                self.video_files.append(self.master_directory + video_file)

        self.all_data = np.empty(shape = (len(self.npy_files),self.width, self.height))

        for i, npy_file in enumerate(self.npy_files):
            data = np.load(npy_file)
            self.all_data[i] = data

        np.save(self.depth_df, self.all_data)

    def select_regions(self):
        background = self.all_data[-1] - self.all_data[0]
        pl.imshow(self.all_data[-1])
        pl.colorbar()
        pl.title('Identify regions with sand')
        self.Tank_ROI = roipoly(roicolor='r')
        np.save(self.tank_mask_file, self.Tank_ROI.getMask(background))

        
        pl.imshow(background)
        pl.colorbar()
        self.Tank_ROI.displayROI()
        pl.title('Identify regions with pit')
        self.Pit_ROI = roipoly(roicolor='b')
        np.save(self.pit_mask, self.Pit_ROI.getMask(background))


        pl.imshow(background)
        pl.colorbar()
        self.Sand_ROI.displayROI()
        self.Pit_ROI.displayROI()
        pl.title('Identify regions with Castle')
        self.Castle_ROI = roipoly(roicolor='g')
        np.save(self.castle_mask, self.Castle_ROI.getMask(background))
        
    def create_heatmap_video(self):

        self.d_min = np.nanpercentile(self.all_data, 0.1)
        self.d_max = np.nanpercentile(self.all_data, 99.9)
        self.d_min2 = np.nanpercentile(self.all_data - self.all_data[0], 0.1)
        self.d_max2 = np.nanpercentile(self.all_data - self.all_data[0], 99.9)
        
        img = cv2.imread(self.jpg_files[0])

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(2,2,1)       
        self.ax2 = self.fig.add_subplot(2,2,2)
        self.ax3 = self.fig.add_subplot(2,2,3)
        self.ax4 = self.fig.add_subplot(2,2,4)

        for axis in [self.ax1, self.ax2, self.ax3, self.ax4]:
            s_l = self.Sand_ROI.ret_line()
            p_l = self.Pit_ROI.ret_line()
            c_l = self.Castle_ROI.ret_line()

            print(axis)
            print(s_l)
            axis.add_line(s_l)
            axis.add_line(p_l)
            axis.add_line(c_l)
 
        seaborn.heatmap(self.all_data[0], ax = self.ax1, vmin = self.d_min, vmax = self.d_max, xticklabels=False, yticklabels=False, cbar = True)
        self.ax2.imshow(img)
        self.ax2.set_axis_off()
        seaborn.heatmap(self.all_data[0]-self.all_data[0], ax = self.ax3, vmin = self.d_min2, vmax = self.d_max2, xticklabels=False, yticklabels=False, cbar = True)
        seaborn.heatmap(self.smooth_data[0]-self.smooth_data[0], ax = self.ax4, vmin = self.d_min2, vmax = self.d_max2, xticklabels=False, yticklabels=False, cbar = True)
        
        writer = animation.writers['ffmpeg'](fps=10, metadata=dict(artist='Patrick McGrath'), bitrate=1800)
        anim = animation.FuncAnimation(self.fig, self.animate, frames = self.all_data.shape[0], repeat = False)
        anim.save(self.summary_video, writer = writer)        

    def animate(self, i):
        
        img = cv2.imread(self.jpg_files[i])

        self.fig.clf()
        self.ax1 = self.fig.add_subplot(2,2,1)       
        self.ax2 = self.fig.add_subplot(2,2,2)
        self.ax3 = self.fig.add_subplot(2,2,3)
        self.ax4 = self.fig.add_subplot(2,2,4)

        for axis in [self.ax1, self.ax2, self.ax3, self.ax4]:
            s_l = self.Sand_ROI.ret_line()
            p_l = self.Pit_ROI.ret_line()
            c_l = self.Castle_ROI.ret_line()

            print(axis)
            print(s_l)
            axis.add_line(s_l)
            axis.add_line(p_l)
            axis.add_line(c_l)
 
        seaborn.heatmap(self.all_data[i], ax = self.ax1, vmin = self.d_min, vmax = self.d_max, xticklabels=False, yticklabels=False, cbar = True)
        self.ax2.imshow(img)
        self.ax2.set_axis_off()
#        self.ax2.set_xticklabels([])
#        self.ax2.set_yticklabels([])
        seaborn.heatmap(self.all_data[i]-self.all_data[0], ax = self.ax3, vmin = self.d_min2, vmax = self.d_max2, xticklabels=False, yticklabels=False, cbar = True)
        seaborn.heatmap(self.smooth_data[i]-self.smooth_data[0], ax = self.ax4, vmin = self.d_min2, vmax = self.d_max2, xticklabels=False, yticklabels=False, cbar = True)

            def smooth_data(self):
        self.smooth_data = np.empty(shape = (len(self.npy_files),) + self.crop_shape)

        for i in range(0,self.all_data.shape[1]):
            print(i)
            for j in range(0,self.all_data.shape[2]):
                self.smooth_data[:,i,j] = scipy.signal.savgol_filter(self.all_data[:,i,j], 51,3)

        np.save(self.smooth_data_file, self.all_data)

    def summarize_data(self):
        t_data = np.load(self.data_file)
        s_background = np.load(self.sand_mask)
        p_background = np.load(self.pit_mask)
        c_background = np.load(self.castle_mask)
        nb_background = np.array(s_background)
        nb_background[p_background == True] = False
        nb_background[c_background == True] = False

        t = []
        s_data = []
        p_data = []
        c_data = []
        nb_data = []
        
        for i in range(0,t_data.shape[0]):
            t.append(i)
            s_data.append(np.nanmean(t_data[i][s_background == True]))
            p_data.append(np.nanmean(t_data[i][p_background == True]))
            c_data.append(np.nanmean(t_data[i][c_background == True]))
            nb_data.append(np.nanmean(t_data[i][nb_background == True]))

        plt.plot(t,s_data, label = 'sand')
        plt.plot(t,p_data, label = 'pit')
        plt.plot(t,c_data, label = 'castle')
        plt.plot(t,nb_data, label = 'sand_not_pit_castle')
        plt.legend()
    
        plt.show()
