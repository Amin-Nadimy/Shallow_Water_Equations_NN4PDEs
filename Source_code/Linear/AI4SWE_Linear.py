'''
 Amin Nadimy, Boyang Chen, Claire Heaney, Christopher Pain
 Department of Earth Science and Engineering
 Imperial College London

 amin.nadimy19@imperial.ac.uk
'''
#-- Import general libraries
import os
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(is_gpu)

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
# create a torch constant
dt = torch.tensor([0.5], device=device)
dx = torch.tensor([5.0], device=device)
nx = torch.tensor([951], device=device)
ny = torch.tensor([611], device=device) 
CFL = np.sqrt(9.81*4)*dt/dx
print('Grid size:', dx,'CFL:', CFL)
# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0], device=device)
# Isotropic Laplacian  
w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)
# Gradient in x 
w2 = torch.tensor([[[[1/(12*dx)], [0.0], [-1/(12*dx)]],
                    [[1/(3*dx)] , [0.0], [-1/(3*dx)]] ,
                    [[1/(12*dx)], [0.0], [-1/(12*dx)]]]], device=device)
# Gradient in y 
w3 = torch.tensor([[[[-1/(12*dx)], [-1/(3*dx)], [-1/(12*dx)]],
                    [[0.0]       , [0.0]      , [0.0]]       ,
                    [[1/(12*dx)] , [1/(3*dx)] , [1/(12*dx)]]]], device=device)
# Consistant mass matrix 
wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                    [[0.11] ,  [0.44], [0.11]],
                    [[0.028], [0.11] , [0.028]]]], device=device)
w1 = torch.reshape(w1,(1,1,3,3))
w2 = torch.reshape(w2,(1,1,3,3))
w3 = torch.reshape(w3,(1,1,3,3))
wm = torch.reshape(wm,(1,1,3,3))
#######################################################
################# Numerical parameters ################
ntime = 489600              # Time steps
n_out = 10000                 # Results output
nrestart = 0                # Last time step for restart
ctime_old = 0               # Last ctime for restart
mgsolver = True             # Multigrid solver for non-hydrostatic pressure 
nsafe = 0.5                 # Continuty equation residuals
ctime = 0                   # Initialise ctime   
save_fig = False             # Save results
Restart = False             # Restart
eplsion_k = 1e-04           # Stablisatin factor in Petrov-Galerkin for velocity
eplsion_eta = 1e-04         # Stablisatin factor in Petrov-Galerkin for height
beta = 4                    # diagonal factor in mass term
real_time = torch.tensor([0.0], device=device)
istep = 0

# # # ################################### # # #
# # # ######   Physical parameters    ### # # #
# # # ################################### # # #
g_x = 0;g_y = 0;g_z = 9.81  # Gravity acceleration (m/s2) 
rho = 1/g_z                 # Resulting density
diag = -np.array(w1.cpu())[0,0,1,1]

# # # ################################### # # #
# # # ######   Create field (tensor)  ### # # #
# # # ################################### # # #
input_shape = (1, 1, ny, nx)
values_u = torch.zeros(input_shape, device=device)
values_v = torch.zeros(input_shape, device=device)
a_u = torch.zeros(input_shape, device=device)
a_v = torch.zeros(input_shape, device=device)
b_u = torch.zeros(input_shape, device=device)
b_v = torch.zeros(input_shape, device=device)
c_u = torch.zeros(input_shape, device=device)
c_v = torch.zeros(input_shape, device=device)
eta1 = torch.zeros(input_shape, device=device)
eta2 = torch.zeros(input_shape, device=device)
values_hh = torch.zeros(input_shape, device=device)
dif_values_h = torch.zeros(input_shape, device=device)
values_h_old = torch.zeros(input_shape, device=device)
sigma_q = torch.zeros(input_shape, device=device)
k_u = torch.zeros(input_shape, device=device)
k_v = torch.zeros(input_shape, device=device)
k_x = torch.zeros(input_shape, device=device)
k_y = torch.zeros(input_shape, device=device)
b = torch.zeros(input_shape, device=device)
# Padding
input_shape_pd = (1, 1, ny + 2, nx + 2)
values_uu = torch.zeros(input_shape_pd, device=device)
values_vv = torch.zeros(input_shape_pd, device=device)
b_uu = torch.zeros(input_shape_pd, device=device)
b_vv = torch.zeros(input_shape_pd, device=device)
eta1_p = torch.zeros(input_shape_pd, device=device)
dif_values_hh = torch.zeros(input_shape_pd, device=device)
values_hhp = torch.zeros(input_shape_pd, device=device)
values_hp = torch.zeros(input_shape_pd, device=device)
k_uu = torch.zeros(input_shape_pd, device=device)
k_vv = torch.zeros(input_shape_pd, device=device)
# stablisation factor
k1 = torch.ones(input_shape, device=device)*eplsion_eta
k2 = torch.zeros(input_shape, device=device)    
k3 = torch.ones(input_shape, device=device)*dx**2*0.5/dt # for transient problem
#######################################################
print('============== Numerical parameters ===============')
print('Mesh resolution:', values_v.shape)
print('Time step:', ntime)
print('Initial time:', ctime)
print('Diagonal componet:', diag)
#######################################################
# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #
# Specify the dimensions of the 2D data (width and height)
# Open the .raw file for reading
values_h = torch.zeros(input_shape, device=device)
values_H = torch.zeros(input_shape, device=device)
with open('carlisle-5m.dem.raw', 'r') as file:
    # Read the entire content of the file and split it into individual values
    data = file.read().split()
# Convert the string values to floats
mesh = np.array([float(value) for value in data[12:]])
# Now, float_values contains a list of floating-point numbers from the .raw file
mesh = mesh.reshape(int(data[3]),int(data[1]))
print(mesh.shape,values_H.shape)
values_H[0,0,:,:] = torch.tensor(mesh, device=device)
# defining the source term
x_origin = 338500 ; y_origin = 554700
df = pd.read_csv('carlisle.bci', delim_whitespace=True)
x_upstream1 = [] ; y_upstream1 = []
x_upstream2 = [] ; y_upstream2 = []
x_upstream3 = [] ; y_upstream3 = []
for index, row in df.iterrows():
    # Check if the 'discharge' column in the current row is 'upstream3'
    if row['discharge'] == 'upstream1':
        # Append the value from the second column to the list
        x_upstream1.append((df['x'][index] - x_origin)//5)
        y_upstream1.append((df['y'][index] - y_origin)//5)
    elif row['discharge'] == 'upstream2':
        x_upstream2.append((df['x'][index] - x_origin)//5)
        y_upstream2.append((df['y'][index] - y_origin)//5)
    elif row['discharge'] == 'upstream3':
        x_upstream3.append((df['x'][index] - x_origin)//5)
        y_upstream3.append((df['y'][index] - y_origin)//5)
print('upstream1:'); print('x:',x_upstream1) ; print('y:',y_upstream1); print('')
print('upstream2:'); print('x:',x_upstream2) ; print('y:',y_upstream2) ; print('')
print('upstream3:') ; print('x:',x_upstream3) ; print('y:',y_upstream3)

source_h = torch.zeros(input_shape, device=device)
df = pd.read_csv('flowrates.csv', delim_whitespace=True)
rate1 = df['upstream1']/5#*0.93
rate2 = df['upstream2']/5#*0.93
rate3 = df['upstream3']/5#*0.93

def get_source(time):
    global rate1, rate2, rate3
    indx = time.item() // 900 #900 is the time interval in the given time series
    for i in range(len(x_upstream1)):
        source_h[0,0,((ny-y_upstream1[i])),x_upstream1[i]] = (rate1[indx+1] - rate1[indx])/900 * (time%900) + rate1[indx]
    for i in range(len(x_upstream2)):
        source_h[0,0,-22,x_upstream2[i]+5]                 = (rate2[indx+1] - rate2[indx])/900 * (time%900) + rate2[indx]
    for i in range(len(x_upstream3)):
        source_h[0,0,-4,x_upstream3[i]]                    = (rate3[indx+1] - rate3[indx])/900 * (time%900) + rate3[indx]

df = pd.read_csv('Point_coor_paper.csv')
x_outflow = (df['x']-x_origin)//dx.cpu().numpy()
y_outflow = (df['y']-y_origin)//dx.cpu().numpy()
names = df['position_name']
x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14,x15,y15=map(int, [x_outflow[0], y_outflow[0], x_outflow[1], y_outflow[1], x_outflow[2], y_outflow[2],
                                                                                                                x_outflow[3], y_outflow[3], x_outflow[4], y_outflow[4], x_outflow[5], y_outflow[5],
                                                                                                                x_outflow[6], y_outflow[6], x_outflow[7], y_outflow[7], x_outflow[8], y_outflow[8],
                                                                                                                x_outflow[9], y_outflow[9], x_outflow[10], y_outflow[10], x_outflow[11], y_outflow[11],
                                                                                                                x_outflow[12], y_outflow[12], x_outflow[13], y_outflow[13], x_outflow[14], y_outflow[14]])
print(x_outflow,y_outflow)
# Transfer array into tensor
values_h = values_H
values_H = - values_H
# # # ################################### # # #
# # # #########   AI4SWE MAIN ########### # # #
# # # ################################### # # #
class AI4SWE(nn.Module):
    """docstring for two_step"""
    def __init__(self):
        super(AI4SWE, self).__init__()
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer

    def boundary_condition_u(self, values_u, values_uu):
        ny = values_u.shape[2]
        nx = values_u.shape[3]
        nny = values_uu.shape[2]
        nnx = values_uu.shape[3]

        values_uu[0,0,1:nny-1,1:nnx-1] = values_u[0,0,:,:]
        values_uu[0,0,:,0] =  values_uu[0,0,:,1]*0 
        values_uu[0,0,:,nx+1] = values_uu[0,0,:,nx]*0
        values_uu[0,0,0,:] = values_uu[0,0,1,:] 
        values_uu[0,0,ny+1,:] = values_uu[0,0,ny,:]
        return values_uu   

    def boundary_condition_v(self, values_v, values_vv):
        ny = values_v.shape[2]
        nx = values_v.shape[3]
        nny = values_vv.shape[2]
        nnx = values_vv.shape[3]

        values_vv[0,0,1:nny-1,1:nnx-1] = values_v[0,0,:,:]
        values_vv[0,0,:,0] =  values_vv[0,0,:,1]
        values_vv[0,0,:,nx+1] = values_vv[0,0,:,nx]
        values_vv[0,0,0,:] = values_vv[0,0,1,:]*0
        values_vv[0,0,ny+1,:] = values_vv[0,0,ny,:]*0
        return values_vv       

    def boundary_condition_eta(self, values_h, values_hp):
        ny = values_h.shape[2]
        nx = values_h.shape[3]
        nny = values_hp.shape[2]
        nnx = values_hp.shape[3]

        values_hp[0,0,1:nny-1,1:nnx-1] = values_h[0,0,:,:]
        values_hp[0,0,:,nx+1] = values_hp[0,0,:,nx]*0      
        values_hp[0,0,:,0] = values_hp[0,0,:,1]*0 
        values_hp[0,0,ny+1,:] = values_hp[0,0,ny,:]*0  
        values_hp[0,0,0,:] = values_hp[0,0,1,:]*0 
        return values_hp   

    def PG_vector(self, values_uu, values_vv, values_u, values_v, k3):
        k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_uu)) / \
            (1e-03  + (torch.abs(self.xadv(values_uu)) * (dx**-2) + torch.abs(self.yadv(values_uu)) * (dx**-2)) / 2)

        k_v = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_vv)) / \
            (1e-03  + (torch.abs(self.xadv(values_vv)) * (dx**-2) + torch.abs(self.yadv(values_vv)) * (dx**-2)) / 2)

        k_uu = F.pad(torch.minimum(k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        k_vv = F.pad(torch.minimum(k_v, k3) , (1, 1, 1, 1), mode='constant', value=0)

        k_x = 0.5 * (k_u * self.diff(values_uu) + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 0.5 * (k_v * self.diff(values_vv) + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        return k_x, k_y

    def PG_scalar(self, values_hh, values_h, values_u, values_v, k3):
        k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_hh)) / \
            (1e-03 + (torch.abs(self.xadv(values_hh)) * (dx**-2) + torch.abs(self.yadv(values_hh)) * (dx**-2)) / 2)  
      
        k_uu = F.pad(torch.minimum(k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        return 0.5 * (k_u * self.diff(values_hh) + self.diff(values_hh * k_uu) - values_h * self.diff(k_uu))        

    def forward(self, values_u, values_uu, values_v, values_vv, values_H, values_h, values_hp, b_u, b_uu, b_v, b_vv, dt, rho, k1, k2, k3, eta1_p, source_h, dif_values_h, dif_values_hh, values_hh, values_hhp):
        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)

        [k_x,k_y] = self.PG_vector(values_uu, values_vv, values_u, values_v, k3)
        b_u = (k_x * dt - values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt) * 0.5 + values_u
        b_v = (k_y * dt - values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt) * 0.5 + values_v
        b_u = b_u - self.xadv(self.boundary_condition_eta(values_h,values_hp)) * dt
        b_v = b_v - self.yadv(self.boundary_condition_eta(values_h,values_hp)) * dt

        b_uu = self.boundary_condition_u(b_u,b_uu)       
        b_vv = self.boundary_condition_v(b_v,b_vv)

        sigma_q = (b_u**2 + b_v**2)**0.5 * 0.055**2 / (torch.maximum( k1,
            dx*self.cmm(self.boundary_condition_eta(values_H+values_h,values_hp))*0.01+(values_H+values_h)*0.99 )**(4/3)) 

        b_u = b_u / (1 + sigma_q * dt / rho)
        b_v = b_v / (1 + sigma_q * dt / rho)

        [k_x,k_y] = self.PG_vector(b_uu, b_vv, b_u, b_v, k3)
        values_u = values_u + k_x * dt - b_u * self.xadv(b_uu) * dt - b_v * self.yadv(b_uu) * dt   
        values_v = values_v + k_y * dt - b_u * self.xadv(b_vv) * dt - b_v * self.yadv(b_vv) * dt 
        values_u = values_u - self.xadv(self.boundary_condition_eta(values_h,values_hp)) * dt
        values_v = values_v - self.yadv(self.boundary_condition_eta(values_h,values_hp)) * dt     

        sigma_q = (values_u**2 + values_v**2)**0.5 * 0.055**2 / (torch.maximum( k1,
            dx*self.cmm(self.boundary_condition_eta(values_H+values_h,values_hp))*0.01+(values_H+values_h)*0.99 )**(4/3))

        values_u = values_u / (1 + sigma_q * dt / rho)
        values_v = values_v / (1 + sigma_q * dt / rho)

        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)
        eta1 = torch.maximum(k2,(values_H+values_h))
        eta2 = torch.maximum(k1,(values_H+values_h))

        b = beta * rho * (-self.xadv(self.boundary_condition_eta(eta1,eta1_p)) * values_u - \
                           self.yadv(self.boundary_condition_eta(eta1,eta1_p)) * values_v - \
                           eta1 * self.xadv(values_uu) - eta1 * self.yadv(values_vv) + \
                           self.PG_scalar(self.boundary_condition_eta(eta1,eta1_p), eta1, values_u, values_v, k3) - \
                           self.cmm(self.boundary_condition_eta(dif_values_h,dif_values_hh)) / dt + source_h) / (dt * eta2)   
        values_h_old = values_h.clone()
        for i in range(2):
            values_hh = values_hh - (-self.diff(self.boundary_condition_eta(values_hh,values_hhp)) + beta * rho / (dt**2 * eta2) * values_hh) / \
                    (diag + beta * rho / (dt**2 * eta2)) + b / (diag + beta * rho / (dt**2 * eta2))
        values_h = values_h + values_hh
        dif_values_h = values_h - values_h_old 
        values_u = values_u - self.xadv(self.boundary_condition_eta(values_hh,values_hhp)) * dt / rho
        values_v = values_v - self.yadv(self.boundary_condition_eta(values_hh,values_hhp)) * dt / rho 

        return values_u, values_v, values_h, values_hh, b, dif_values_h, sigma_q

model = AI4SWE().to(device)

file_name = 'AI4SWE.csv'
with open(file_name, 'w', newline='') as csvfile:
    fieldnames = ['time_s', 'Sheepmount_h', 'Bothcherby_h', 'Denton_Holme_h', 'Building_1_h', 'Water_mark_3_h', 'Water_mark_4_h', 'Water_mark_5_h', 'Water_mark_6_h', 'Water_mark_7_h', 'Bus_depot_h',
                            'Substation_h', 'Pallett_Yard_h', 'Brown_Bros_h', 'Water_mark_1_h', 'Water_mark_2_h', 'Sheepmount_u', 'Bothcherby_u', 'Denton_Holme_u', 'Building_1_u', 'Water_mark_3_u', 'Water_mark_4_u', 
                            'Water_mark_5_u', 'Water_mark_6_u', 'Water_mark_7_u', 'Bus_depot_u', 'Substation_u', 'Pallett_Yard_u', 'Brown_Bros_u', 'Water_mark_1_u', 'Water_mark_2_u', 'tot_vol']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()    # Write the header


start = time.time()
with torch.no_grad():

    for itime in range(1,ntime+1):
        get_source(real_time)
        for t in range(2):
            [values_u, values_v, values_h, values_hh, b, dif_values_h, sigma_q] = model(values_u, values_uu, values_v, values_vv, values_H, values_h, 
                    values_hp, b_u, b_uu, b_v, b_vv, dt, rho, k1, k2, k3, eta1_p, source_h, dif_values_h, dif_values_hh, values_hh, values_hhp)
# output          
        real_time = real_time + dt
        istep +=1

        if np.max(np.abs(values_hh.cpu().detach().numpy())) > 10.0:
            print('Not converged !!!!!!')
            break
        print('Time step:', itime, 'height correction:', np.max(values_hh.cpu().detach().numpy())) 

        if save_fig == True and itime % n_out == 0:
            np.save("H"+str(itime), arr=values_H.cpu().detach().numpy()[0,0,:,:])
            np.save("h"+str(itime), arr=values_h.cpu().detach().numpy()[0,0,:,:])
            np.save("u"+str(itime), arr=values_v.cpu().detach().numpy()[0,0,:,:])
            np.save("v"+str(itime), arr=values_u.cpu().detach().numpy()[0,0,:,:])

        if (istep*dt) % 900 <= 0.001:
            print(f'Time step: {itime}, {istep}, img = {int(real_time.item()/900)} , time in seconds = {real_time.item():.0f}')
                # Open the CSV file for appending the data
            with open(file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    # Write the data for each iteration
                writer.writerow({'time_s':'{:.0f}'.format(real_time.item()), 
                                     'Sheepmount_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y1,x1].item())   , 'Bothcherby_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y2,x2].item()),
                                     'Denton_Holme_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y3,x3].item())   , 'Building_1_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y4,x4].item()),
                                     'Water_mark_3_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y5,x5].item())   , 'Water_mark_4_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y6,x6].item()),
                                     'Water_mark_5_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y7,x7].item())   , 'Water_mark_6_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y8,x8].item()),
                                     'Water_mark_7_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y9,x9].item())   , 'Bus_depot_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y10-19,x10+3].item()),
                                     'Substation_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y11,x11].item()), 'Pallett_Yard_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y12,x12+5].item()),
                                     'Brown_Bros_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y13,x13].item()), 'Water_mark_1_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y14-12,x14-13].item()),
                                     'Water_mark_2_h':'{:.1f}'.format((values_h+values_H)[0,0,611-y15-9,x15-4].item()),
                                     'Sheepmount_u':'{:.1f}'.format(values_u[0,0,611-y1,x1].item()), 'Bothcherby_u':'{:.1f}'.format(values_u[0,0,611-y2,x2].item()),
                                     'Denton_Holme_u':'{:.1f}'.format(values_u[0,0,611-y3,x3].item()), 'Building_1_u':'{:.1f}'.format(values_u[0,0,611-y4,x4].item()),
                                     'Water_mark_3_u':'{:.1f}'.format(values_u[0,0,611-y5,x5].item()), 'Water_mark_4_u':'{:.1f}'.format(values_u[0,0,611-y6,x6].item()),
                                     'Water_mark_5_u':'{:.1f}'.format(values_u[0,0,611-y7,x7].item()), 'Water_mark_6_u':'{:.1f}'.format(values_u[0,0,611-y8,x8].item()),
                                     'Water_mark_7_u':'{:.1f}'.format(values_u[0,0,611-y9,x9].item()), 'Bus_depot_u':'{:.1f}'.format(values_u[0,0,611-y10,x10].item()),
                                     'Substation_u':'{:.1f}'.format(values_u[0,0,611-y11,x11].item()), 'Pallett_Yard_u':'{:.1f}'.format(values_u[0,0,611-y12,x12].item()),
                                     'Brown_Bros_u':'{:.1f}'.format(values_u[0,0,611-y13,x13].item()), 'Water_mark_1_u':'{:.1f}'.format(values_u[0,0,611-y14,x14].item()),
                                     'Water_mark_2_u':'{:.1f}'.format(values_u[0,0,611-y15,x15].item()),
                                     'tot_vol':'{:.1f}'.format(25*torch.sum((values_h+values_H)[0,0,:,:]).item())})


    end = time.time()
    print('time',(end-start))
