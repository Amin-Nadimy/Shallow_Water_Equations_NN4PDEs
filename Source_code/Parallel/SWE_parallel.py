import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.multiprocessing import Process, set_start_method
import SWE_2D_diff_res_Linear as an
import torch.distributed as dist
import os
import subprocess


################# Numerical parameters ################
ntime = 36000              # Time steps
n_out = 100                 # Results output
nrestart = 0                # Last time step for restart
ctime_old = 0               # Last ctime for restart
mgsolver = True             # Multigrid solver for non-hydrostatic pressure
nsafe = 0.5                 # Continuty equation residuals
ctime = 0                   # Initialise ctime
save_fig = True             # Save results
Restart = False             # Restart
eplsion_k = 1e-04          # Stablisatin factor in Petrov-Galerkin for velocity
eplsion_eta = 1e-04         # Stablisatin factor in Petrov-Galerkin for height
beta = 4                    # diagonal factor in mass term
real_time = 0
istep = 0
manning = 0.055
dt = 0.5
################# Physical parameters #################
g_x = 0;g_y = 0;g_z = 9.81  # Gravity acceleration (m/s2)
rho = 1/g_z                 # Resulting density

global_nx = 952
global_ny = 612
no_domains_x = 2
no_domains_y = 2

sd_indx_with_lower_res = [] #also change the rate1 rte2 and rate3
sd_indx_with_2x_res = {} #{4:2}



path = '/home/an619/Desktop/git/AI/RMS/'  # workstation
# path = '/home/amin/Desktop/AI/RMS/'       # laptop

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    num_gpu_devices = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(num_gpu_devices)]

    print(f"Number of available GPU devices: {num_gpu_devices}")
    device = []
    for i, device_name in enumerate(device_names):
        device.append(torch.device(f"cuda:{i}"))
        print(f"GPU {i}: {device_name}, {device[i]}")
        
else:
    device = 'cpu'
    print("No GPU devices available. Using CPU.")
is_gpu = torch.cuda.is_available()


# Set environment variable
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # Enable XLA devices

# Check Nvidia GPU details
nvidia_smi_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(nvidia_smi_output.stdout)

# Check CPU details
lscpu_output = subprocess.run(['lscpu'], capture_output=True, text=True)
print(lscpu_output.stdout)

# Check memory details
free_output = subprocess.run(['free', '-h', '--si'], capture_output=True, text=True)
mem_info = [line for line in free_output.stdout.split('\n') if 'Mem:' in line]
if mem_info:
    print(mem_info[0].split()[1])


# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #
# Open the .raw file for reading
with open(f'{path}carlisle-5m.dem.raw', 'r') as file:
    # Read the entire content of the file and split it into individual values
    data = file.read().split()

# Convert the string values to floats
mesh = torch.tensor([float(value) for value in data[12:]])

# Now, float_values contains a list of floating-point numbers from the .raw file

mesh = mesh.reshape(int(data[3]),int(data[1]))
# make the size even numbers (left, right, top, bottom) padding
mesh = F.pad(mesh, (0,1,0,1), mode='constant', value=0)
mesh[:,-1] = mesh[:,-2]
mesh[-1,:] = mesh[-2,:]



def gen_subdomains(dt, path, mesh, global_nx, global_ny, no_domains_x, no_domains_y, sd_indx_with_lower_res, sd_indx_with_2x_res):
    no_domains = no_domains_x * no_domains_y

    sub_nx = global_nx//no_domains_x
    sub_ny = global_ny//no_domains_y

    print('sub_nx, sub_ny :', sub_nx, sub_ny)

    nx = [sub_nx]*no_domains
    ny = [sub_ny]*no_domains
    dx = [5]*no_domains

    ratios = [0.5 if i in sd_indx_with_lower_res else 1 for i in range(no_domains)]

    dx = [val/ratios[i] for i, val in enumerate(dx)]
    nx = [int(val/np.floor_divide(1,ratios[i])) for i, val in enumerate(nx)]
    ny = [int(val/np.floor_divide(1,ratios[i])) for i, val in enumerate(ny)]

    # defining domains with 2x resolution and set dx, dy ,nx and ny
    global_ratio = ratios
    for key, value in sd_indx_with_2x_res.items():
        dx[key] /= value
        nx[key] *= value
        ny[key] *= value
        global_ratio[key] = value
        
    dy = dx

    sd = [an.subdomain_2D(nx[i], ny[i], dx[i], dy[i] , dt, eplsion_eta, i) for i in range(no_domains)]

    # Set the inner neigs correctly
    an.set_face_neigs(sd, no_domains_x, no_domains_y)

    # set the corner_neig
    an.set_corner_neighbors(sd, no_domains)
    print('ratios:', ratios)
    print('nx    :', nx)
    print('dx    :', dx)
    print('These 2numbers must match the initial mesh size', sub_nx*no_domains_x, sub_ny*no_domains_y)

    def create_halos(sd):
        for ele in range(no_domains):
            # Get the size of the block along the x and y axes
            ny, nx = sd[ele].values_u.shape[2], sd[ele].values_u.shape[3]

            # Create a list where each element is a tensor corresponding to a face of the block
            sd[ele].halo_u = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face
            
            sd[ele].halo_v = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face
            
            sd[ele].halo_b_u = [torch.zeros((nx,)),  # Bottom face
                                torch.zeros((ny,)),  # Left face
                                torch.zeros((ny,)),  # Right face
                                torch.zeros((nx,))]  # Top face
            
            sd[ele].halo_b_v = [torch.zeros((nx,)),  # Bottom face
                                torch.zeros((ny,)),  # Left face
                                torch.zeros((ny,)),  # Right face
                                torch.zeros((nx,))]  # Top face
            
            sd[ele].halo_h = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face
            
            sd[ele].halo_hh = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face
            
            sd[ele].halo_dif_h = [torch.zeros((nx,)),  # Bottom face
                                torch.zeros((ny,)),  # Left face
                                torch.zeros((ny,)),  # Right face
                                torch.zeros((nx,))]  # Top face
            
            sd[ele].halo_eta = [torch.zeros((nx,)),  # Bottom face
                                torch.zeros((ny,)),  # Left face
                                torch.zeros((ny,)),  # Right face
                                torch.zeros((nx,))]  # Top face
            
            sd[ele].halo_eta1 = [torch.zeros((nx,)),  # Bottom face
                                torch.zeros((ny,)),  # Left face
                                torch.zeros((ny,)),  # Right face
                                torch.zeros((nx,))]  # Top face

        return

    # Create halos for each block :: sd[ele].halo_u[iface]
    create_halos(sd)

    # # # ################################### # # #
    # # # ######   Numerial parameters ###### # # #
    # # # ################################### # # #
    CFL = np.sqrt(9.81*4)*dt/sd[0].dx
    print('Grid size:', sd[0].dx,'CFL:', CFL)


    #  plotting the domain with boundaries
    # Calculate the size of each subdomain
    subdomain_size_x = mesh.shape[1] // no_domains_x
    subdomain_size_y = mesh.shape[0] // no_domains_y

    # Plot the mesh
    # plt.imshow(mesh)

    # # Draw the boundaries of the subdomains
    # for i in range(1, no_domains_x):
    #     plt.axvline(i * subdomain_size_x, color='w')
    # for j in range(1, no_domains_y):
    #     plt.axhline(j * subdomain_size_y, color='w')

    # plt.close()


    def split_mesh_into_subdomains(tensor, num_blocks_y, num_blocks_x):
        '''
        split the mesh into tiles and gives coordinates and indices of each subdomain
        subdomain numbering now starts from bottom left and is row major
        '''
        # Split the tensor into chunks along the x axis
        chunks_x = tensor.chunk(num_blocks_x, dim=0)
        
        # Initialize lists to store blocks, coordinates, and corresponding indices
        blocks = []
        coordinates = []
        indices = []

        # Loop through chunks along the x axis in reverse order
        for i, chunk_x in enumerate(reversed(chunks_x)):
            # Split each chunk along the y axis
            chunks_y = chunk_x.chunk(num_blocks_y, dim=1)
            
            # Loop through chunks along the y axis
            for j, chunk_y in enumerate(chunks_y):
                # Append block, coordinates, and corresponding indices
                blocks.append(chunk_y)
                coordinates.append([i, j])
                index = i * num_blocks_y + j  # Calculate the overall subdomain index
                indices.append(index)
                
        return blocks, coordinates, indices
    tiles, subdomain_coordinates, subdomain_indices = split_mesh_into_subdomains(mesh, no_domains_x, no_domains_y)


    # change the resolution of subdomains
    def scale_2D_subdomains(tile, ratio):
        '''
        Scales a 2D tensor by a given ratio. The tensor is assumed to have a shape
        of (height, width).
        '''
        if ratio == 1:
            return tile
        
        elif ratio < 1:
            # Reshape the tensor to 4D, with the last two dimensions being 2x2
            unfolded = tile.unfold(0, 2, 2).unfold(1, 2, 2)

            # Compute the mean over the last two dimensions
            tile = unfolded.mean(dim=[2, 3])
        elif ratio == 2:
            # Upscale the tensor
            # First, add an extra dimension for channels and batch size
            tile = tile.unsqueeze(0).unsqueeze(0)
            
            # Then, use interpolate to upscale
            tile = F.interpolate(tile, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Finally, remove the extra dimensions
            tile = tile.squeeze(0).squeeze(0)
        return tile

    # copy tiles into sd.values_H
    for i in range(no_domains):
        tiles[i] = scale_2D_subdomains(tiles[i], global_ratio[i])
        sd[i].values_H[0,0,:,:] = tiles[i]

    for ele in range(no_domains):
        sd[ele].values_h = sd[ele].values_H
        sd[ele].values_hp[0,0,1:-1,1:-1] = sd[ele].values_H
        sd[ele].values_H = -sd[ele].values_H


    x_origin = 338500 ; y_origin = 554700

    df = pd.read_csv(f'{path}T07/carlisle.bci', delim_whitespace=True)

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


    # plt.scatter(x_upstream1, y_upstream1, label='upstream 1', color='red', marker='o')
    # plt.scatter(x_upstream2, y_upstream2, label='upstream 2', color='blue', marker='x')
    # plt.scatter(x_upstream3, y_upstream3, label='upstream 3', color='green', marker='s')
    # plt.legend()
    # plt.xlim(0,len(mesh[0,:]))

    print('upstream1:'); print('x:',x_upstream1) ; print('y:',y_upstream1); print('')
    print('upstream2:'); print('x:',x_upstream2) ; print('y:',y_upstream2) ; print('')
    print('upstream3:') ; print('x:',x_upstream3) ; print('y:',y_upstream3)



    # subdonain_idx_for_source stores the subdomain indices where each source locates.
    subdonain_idx_for_source = []
    def get_point_partition_coordinates(x_list, y_list, no_domains_x, no_domains_y, mesh, subdomain_indices, ratios):
        '''
        Calculating the new point coor based on partitions and resolution of subdomains
        the partition coordinates  and index of a list of points
        :param x_list & y_list: list of x & y coordinates
        :param mesh_size_x: global mesh size in the x direction
        :return: list of partition coordinates and the subdomain index
        '''
        mesh_size_x = mesh.size(1) / no_domains_x
        mesh_size_y = mesh.size(0) / no_domains_y
        point_partition_coordinates = []

        for x, y in zip(x_list, y_list):
            # Calculate the indices of the partition in the x and y directions
            partition_x = int(x / mesh_size_x)
            partition_y = int(y / mesh_size_y)

            point_partition_coordinates.append([partition_x, partition_y])
            # Calculate the overall subdomain index
            subdomain_index = partition_x + partition_y * no_domains_x
        
        subdonain_idx_for_source.append(subdomain_index)
        # apply subdomain position to coor
        x_list = [x-point_partition_coordinates[0][0]*mesh_size_x for x in x_list]
        y_list = [y-point_partition_coordinates[0][1]*mesh_size_y for y in y_list]
        
        # apply resolution change to the coor
        factor = ratios[subdomain_index]
        
        x_list = [int(x*factor) for x in x_list]
        y_list = [int(y*factor) for y in y_list]

        return x_list, y_list, point_partition_coordinates, subdomain_index

    x_upstream1, y_upstream1, par1, sub_idx1 = get_point_partition_coordinates(x_upstream1, y_upstream1, no_domains_x, no_domains_y, mesh, subdomain_indices, ratios)
    x_upstream2, y_upstream2, par2, sub_idx2 = get_point_partition_coordinates(x_upstream2, y_upstream2, no_domains_x, no_domains_y, mesh, subdomain_indices, ratios)
    x_upstream3, y_upstream3, par3, sub_idx3 = get_point_partition_coordinates(x_upstream3, y_upstream3, no_domains_x, no_domains_y, mesh, subdomain_indices, ratios)
    print('upstream1:'); print('x:',x_upstream1) ; print('y:',y_upstream1); print(par1[0][0]); print('')
    print('upstream2:'); print('x:',x_upstream2) ; print('y:',y_upstream2) ; print('')
    print('upstream3:') ; print('x:',x_upstream3) ; print('y:',y_upstream3)

    return dx, subdonain_idx_for_source, x_upstream1, x_upstream2, x_upstream3, y_upstream1, y_upstream2, y_upstream3, ratios, [sd[0], sd[1], sd[2], sd[3]]



dx, subdonain_idx_for_source, x_upstream1, x_upstream2, x_upstream3, y_upstream1, y_upstream2, y_upstream3, ratios, sd_list = gen_subdomains(dt, path, mesh, global_nx, global_ny, no_domains_x, no_domains_y, sd_indx_with_lower_res, sd_indx_with_2x_res)

sd_sources_toBeUpdated = [sd_list[subdonain_idx_for_source[0]].rank,sd_list[subdonain_idx_for_source[1]].rank,sd_list[subdonain_idx_for_source[2]].rank]
df = pd.read_csv(f'{path}T07/flowrates.csv', delim_whitespace=True)
rate1 = df['upstream1']/5
rate2 = df['upstream2']/5
rate3 = df['upstream3']/5
# set the source based on variable time compatible with the real data
def get_source(sd, time):
    global rate1, rate2, rate3, sd_sources_toBeUpdated
    
    indx = time // 900 # 900 is the time interval in the given time series
    if sd.rank == sd_sources_toBeUpdated[0]:
        for i in range(len(x_upstream1)):
            sd.source_h[0,0,((sd.ny-y_upstream1[i])),x_upstream1[i]] = (rate1[indx+1] - rate1[indx])/900 * (time%900) + rate1[indx]
    if sd.rank == sd_sources_toBeUpdated[1]:
        for i in range(len(x_upstream2)):
            sd.source_h[0,0,-int(22*ratios[sd.rank]),x_upstream2[i]] = (rate2[indx+1] - rate2[indx])/900 * (time%900) + rate2[indx]
    if sd.rank == sd_sources_toBeUpdated[2]:
        for i in range(len(x_upstream3)):
            sd.source_h[0,0,-int(4*ratios[sd.rank]),x_upstream3[i]]  = (rate3[indx+1] - rate3[indx])/900 * (time%900) + rate3[indx]
    # if sd.rank == 0 :
    #     sd.source_h[0,0,3,:] = rate2[indx]
    return


def init_process(cuda_indx):
    torch.cuda.set_device(cuda_indx)
    """Initialisation"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12369'
    # Initialisation
    dist.init_process_group(backend='nccl', init_method='env://', world_size=2, rank=cuda_indx)

def scale_it(val1, val2):
        '''
        val1 :: info comming from
        val2 :: info going to
        ratio = val1/val2 size
        '''
        size_val2 = val2.size(0)
        ratio = val1.size(0) / size_val2

        if ratio == 1.0:
            return val1
        elif ratio == 0.5:
            # Use repeat_interleave to increase the size of val1 to match the size of val2
            repeat_factor = size_val2 // val1.size(0)
            val2 = val1.repeat_interleave(repeat_factor)
            return val2
        elif ratio == 2:
            # Calculate the number of elements in each group
            group_size = val1.size(0) // size_val2
            # Reshape val1 into a 2D tensor with size_val2 rows
            val1_reshaped = val1.view(size_val2, -1)
            # Calculate the mean of each row
            val2 = val1_reshaped.mean(dim=1)
            return val2
        elif ratio == 0.25:
            # Repeat each element of val1 4 times
            val2 = val1.repeat_interleave(4)
            return val2
        elif ratio == 4:
            # Reshape val1 into a 2D tensor with size_val2 rows and 4 columns
            val1_reshaped = val1.view(size_val2, -1)
            # Calculate the mean of each row
            val2 = val1_reshaped.mean(dim=1)
            return val2

# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0])
# Isotropic Laplacian
w1 = torch.tensor([[[[1/3], [ 1/3] , [1/3]],
                    [[1/3], [-8/3] , [1/3]],
                    [[1/3], [ 1/3] , [1/3]]]])
# Gradient in x
w2 = torch.tensor([[[[1/12], [0.0], [-1/12]],
                    [[1/3] , [0.0], [-1/3]] ,
                    [[1/12], [0.0], [-1/12]]]])
# Gradient in y
w3 = torch.tensor([[[[-1/12], [-1/3], [-1/12]],
                    [[0.0]  , [0.0] , [0.0]]  ,
                    [[1/12] , [1/3] , [1/12]]]])
# Consistant mass matrix
wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                    [[0.11] , [0.44] , [0.11]],
                    [[0.028], [0.11] , [0.028]]]])
w1 = torch.reshape(w1,(1,1,3,3))
w2 = torch.reshape(w2,(1,1,3,3))
w3 = torch.reshape(w3,(1,1,3,3))
wm = torch.reshape(wm,(1,1,3,3))


# # # ################################### # # #
# # # #########   AI4SWE MAIN ########### # # #
# # # ################################### # # #
class AI4SWE(nn.Module):
    """docstring for two_step"""
    def __init__(self):
        super(AI4SWE, self).__init__()
        self.cmm = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm.weight.data = wm
        
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1 / dx[0]**2
        self.xadv.weight.data = w2 / dx[0]
        self.yadv.weight.data = w3 / dx[0]

        self.diag = -w1[0, 0, 1, 1].item() / (dx[0])**2
            
        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer


    def boundary_condition_u(self, values_u, values_uu, sd):
        values_uu[0,0,1:-1,1:-1] = values_u[0,0, :, :]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            values_uu[0,0, -1, :] = values_uu[0,0, -2, :]
        else:
            values_uu[0,0, -1, 1:-1] = sd.halo_u[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int):
            values_uu[0,0,    :, 0].fill_(0) 
        else:
            values_uu[0,0, 1:-1, 0] = sd.halo_u[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int):
            values_uu[0,0,    :, -1].fill_(0)
        else:
            values_uu[0,0, 1:-1, -1] = sd.halo_u[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            values_uu[0,0, 0, :] = values_uu[0,0, 1, :]
        else:
            values_uu[0,0, 0, 1:-1] = sd.halo_u[3]
            
        return


    def boundary_condition_v(self, values_v, values_vv, sd):
        values_vv[0,0,1:-1,1:-1] = values_v[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            values_vv[0,0, -1,    :].fill_(0)
        else:
            values_vv[0,0, -1, 1:-1] = sd.halo_v[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int):
            values_vv[0,0,      :,   0] = values_vv[0,0, :, 1]
        else:
            values_vv[0,0,   1:-1,   0] = sd.halo_v[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int):   
            values_vv[0,0,      :, -1] = values_vv[0,0, :,-2] 
        else:
            values_vv[0,0,   1:-1, -1] = sd.halo_v[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            values_vv[0,0, 0,    :].fill_(0)
        else:
            values_vv[0,0, 0, 1:-1] = sd.halo_v[3]

        return
    
    
    def boundary_condition_b_u(self, b_u, b_uu, sd):
        b_uu[0,0,1:-1,1:-1] = b_u[0,0, :, :]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            b_uu[0,0, -1,    :] = b_uu[0,0,-2, :]
        else:
            b_uu[0,0, -1, 1:-1] = sd.halo_b_u[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int):
            b_uu[0,0,   :, 0].fill_(0)
        else:
            b_uu[0,0,1:-1, 0] = sd.halo_b_u[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int):
            b_uu[0,0,    :, -1].fill_(0) 
        else:
            b_uu[0,0, 1:-1, -1] = sd.halo_b_u[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            b_uu[0,0, 0,    :] = b_uu[0,0, 1, :]
        else:
            b_uu[0,0, 0, 1:-1] = sd.halo_b_u[3]

        return


    def boundary_condition_b_v(self, b_v, b_vv, sd):
        b_vv[0,0,1:-1,1:-1] = b_v[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            b_vv[0,0, -1,    :].fill_(0)
        else:
            b_vv[0,0, -1, 1:-1] = sd.halo_b_v[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int):
            b_vv[0,0,      :,   0] = b_vv[0,0, :, 1]
        else:
            b_vv[0,0,   1:-1,   0] = sd.halo_b_v[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int):   
            b_vv[0,0,      :, -1] = b_vv[0,0, :,-2]
        else:
            b_vv[0,0,   1:-1, -1] = sd.halo_b_v[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            b_vv[0,0, 0,    :].fill_(0)         # top
        else:
            b_vv[0,0, 0, 1:-1] = sd.halo_b_v[3]

        return


    def boundary_condition_h(self, values_h, values_hp, sd): # Amin:: delete nny etc instead use -1 # use tensor.add()
        values_hp[0,0,1:-1,1:-1] = values_h[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            values_hp[0,0,-1,:].fill_(0)
        else:
            values_hp[0,0,-1,1:-1] = sd.halo_h[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int):
            values_hp[0,0,   :,0].fill_(0)
        else:
            values_hp[0,0,1:-1,0] = sd.halo_h[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int):
            values_hp[0,0,   :,-1].fill_(0)
        else:
            values_hp[0,0,1:-1,-1] = sd.halo_h[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            values_hp[0,0,0,    :].fill_(0)
        else:
            values_hp[0,0,0, 1:-1] = sd.halo_h[3]
                
        return values_hp
        
    
    def boundary_condition_hh(self, values_hh, values_hhp, sd): # Amin:: delete nny etc instead use -1 # use tensor.add()
        values_hhp[0,0,1:-1,1:-1] = (values_hh)[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            values_hhp[0,0,-1,:].fill_(0)
        else:
            values_hhp[0,0,-1, 1:-1] = sd.halo_hh[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int):
            values_hhp[0,0,   :,0].fill_(0)
        else:
            values_hhp[0,0,1:-1,0] = sd.halo_hh[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int): # on the physical boundary
            values_hhp[0,0,   :,-1].fill_(0)
        else:
            values_hhp[0,0,1:-1,-1] = sd.halo_hh[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            values_hhp[0,0,0,:].fill_(0)
        else:
            values_hhp[0,0,0,1:-1] = sd.halo_hh[3]
            
        return values_hhp
    
    
    def boundary_condition_dif_h(self, dif_values_h, dif_values_hh, sd):
        dif_values_hh[0,0,1:-1,1:-1] = (dif_values_h)[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            dif_values_hh[0,0,-1,:].fill_(0)
        else:
            dif_values_hh[0,0,-1,1:-1] = sd.halo_dif_h[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int):
            dif_values_hh[0,0,   :,0].fill_(0)
        else:
            dif_values_hh[0,0,1:-1,0] = sd.halo_dif_h[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int): # on the physical boundary
            dif_values_hh[0,0,   :,-1].fill_(0)
        else:
            dif_values_hh[0,0,1:-1,-1] = sd.halo_dif_h[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            dif_values_hh[0,0,0,:].fill_(0)
        else:
            dif_values_hh[0,0,0,1:-1] = sd.halo_dif_h[3]
        
            
        return dif_values_hh
                                                     
        
    def boundary_condition_eta(self, eta, values_hp, sd):
        values_hp[0,0,1:-1,1:-1] = eta[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            values_hp[0,0,-1,    :].fill_(0)
        else:
            values_hp[0,0,-1, 1:-1] = sd.halo_eta[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int):
            values_hp[0,0,    :,   0].fill_(0)             
        else:
            values_hp[0,0, 1:-1,   0] = sd.halo_eta[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int):
            values_hp[0,0,    :, -1].fill_(0)              
        else:
            values_hp[0,0, 1:-1, -1] = sd.halo_eta[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            values_hp[0,0,0,    :].fill_(0)
        else:
            values_hp[0,0,0, 1:-1] = sd.halo_eta[3]

        return values_hp

    
    def boundary_condition_eta1(self, eta1, eat1_p, sd): # Amin:: delete nny etc instead use -1 # use tensor.add()# check is dimension applys inside the ()
        eat1_p[0,0,1:-1,1:-1] = eta1[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[0], int):
            eat1_p[0,0,-1,    :].fill_(0)
        else:
            eat1_p[0,0,-1, 1:-1] = sd.halo_eta1[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig[1], int): # on the physical boundary
            eat1_p[0,0,    :,0].fill_(0)
        else:
            eat1_p[0,0, 1:-1,0] = sd.halo_eta1[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig[2], int): # on the physical boundary
            eat1_p[0,0,   :,-1].fill_(0)
        else:
            eat1_p[0,0,1:-1,-1] = sd.halo_eta1[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig[3], int):
            eat1_p[0,0,0,    :].fill_(0)
        else:
            eat1_p[0,0,0, 1:-1] = sd.halo_eta1[3]
        
        return eat1_p
    
        
    def update_halo_k_uu(self, sd):
        # --------------------------------------------------------------------------------- bottom
        neig0 = sd.neig[0]
        if not isinstance(neig0, int):
            sd.k_uu[0,0, -1, 1:-1] = sd.halo_k_uu[0]
        # --------------------------------------------------------------------------------- left
        neig1 = sd.neig[1]
        if not isinstance(neig1, int): # has neighbour
            sd.k_uu[0,0, 1:-1, 0] = sd.halo_k_uu[1]
        # --------------------------------------------------------------------------------- right
        neig2 = sd.neig[2]
        if not isinstance(neig2, int):
            sd.k_uu[0,0, 1:-1, -1] = sd.halo_k_uu[2]
        # --------------------------------------------------------------------------------- top
        neig3 = sd.neig[3]
        if not isinstance(neig3, int):
            sd.k_uu[0,0, 0, 1:-1] = sd.halo_k_uu[3]
        return
 

    def update_halo_k_vv(self, sd):
        # --------------------------------------------------------------------------------- bottom
        neig0 = sd.neig[0]
        if not isinstance(neig0, int):
            sd.k_vv[0,0, -1, 1:-1] = sd.halo_k_vv[0]
        # --------------------------------------------------------------------------------- left
        neig1 = sd.neig[1]
        if not isinstance(neig1, int): # has neighbour
            sd.k_vv[0,0,1:-1,0] = sd.halo_k_vv[1]
        # --------------------------------------------------------------------------------- right
        neig2 = sd.neig[2]
        if not isinstance(neig2, int): # has neighbour
            sd.k_vv[0,0,1:-1,-1] = sd.halo_k_vv[2]
        # --------------------------------------------------------------------------------- top
        neig3 = sd.neig[3]
        if not isinstance(neig3, int):
            sd.k_vv[0,0, 0, 1:-1] = sd.halo_k_vv[3]
        return
    
    
    def PG_vector(self, values_uu, values_vv, values_u, values_v, k3, dx, sd):
        sd.k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_uu)) / (1e-03  + (torch.abs(self.xadv(values_uu)) * (dx**-2) + torch.abs(self.yadv(values_uu)) * (dx**-2)) / 2)
        sd.k_v = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_vv)) / (1e-03  + (torch.abs(self.xadv(values_vv)) * (dx**-2) + torch.abs(self.yadv(values_vv)) * (dx**-2)) / 2)

        sd.k_uu = F.pad(torch.minimum(sd.k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        sd.k_vv = F.pad(torch.minimum(sd.k_v, k3) , (1, 1, 1, 1), mode='constant', value=0)
        
        self.update_halo_k_uu(sd)
        self.update_halo_k_vv(sd)
        
        sd.k_x = 0.5 * (sd.k_u * self.diff(values_uu) + self.diff(values_uu * sd.k_uu) - values_u * self.diff(sd.k_uu))
        sd.k_y = 0.5 * (sd.k_v * self.diff(values_vv) + self.diff(values_vv * sd.k_vv) - values_v * self.diff(sd.k_vv))
        return sd.k_x, sd.k_y


    def PG_scalar(self, sd, eta1_p, eta1, values_u, values_v, k3, dx):
        sd.k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(eta1_p)) / (1e-03 + (torch.abs(self.xadv(eta1_p)) * (dx**-2) + torch.abs(self.yadv(eta1_p)) * (dx**-2)) / 2)
        sd.k_uu = F.pad(torch.minimum(sd.k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        self.update_halo_k_uu(sd)
        return 0.5 * (sd.k_u * self.diff(eta1_p) + self.diff(eta1_p * sd.k_uu) - eta1 * self.diff(sd.k_uu))

    def forward(self, values_u, values_v, values_uu, values_vv, values_H, values_h, values_hh, values_hp, values_hhp, dif_values_h, dif_values_hh, source_h, k_x, k_y, sigma_q, eta1, eta2, eta1_p, k1, k2, k3, dx, b, sd, dt, rho):
        self.boundary_condition_u(values_u,values_uu, sd)
        self.boundary_condition_v(values_v,values_vv, sd)
        # ------------------------------------------------------------------------------------------------------------------- 1 step scheme
        k_x, k_y = self.PG_vector(values_uu, values_vv, values_u, values_v, k3, dx, sd)
        
#         b_u = (k_x * dt - values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt) * 0.5 + values_u
#         b_v = (k_y * dt - values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt) * 0.5 + values_v
#         b_u = b_u - self.xadv(self.boundary_condition_h(values_h,values_hp, sd)) * dt
#         b_v = b_v - self.yadv(self.boundary_condition_h(values_h,values_hp, sd)) * dt

#         self.boundary_condition_b_u(b_u,sd.b_uu, sd)
#         self.boundary_condition_b_v(b_v,sd.b_vv, sd)

#         sigma_q = (b_u**2 + b_v**2)**0.5 * 0.055**2 / (torch.maximum( sd.k1,
#             dx*self.cmm(self.boundary_condition_eta(values_H+values_h,values_hp, sd))*0.01+(values_H+values_h)*0.99 )**(4/3))

#         b_u = b_u / (1 + sigma_q * dt / rho)
#         b_v = b_v / (1 + sigma_q * dt / rho)

#         k_x, k_y = self.PG_vector(sd.b_uu, sd.b_vv, b_u, b_v, k3, dx, sd)
            
        values_u = (k_x * dt - values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt) * 0.5 + values_u
        values_v = (k_y * dt - values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt) * 0.5 + values_v
        values_u = values_u - self.xadv(self.boundary_condition_h(values_h,values_hp, sd)) * dt
        values_v = values_v - self.yadv(self.boundary_condition_h(values_h,values_hp, sd)) * dt

        sigma_q = (values_u**2 + values_v**2)**0.5 * 0.055**2 / (torch.maximum( sd.k1,
           dx*self.cmm(self.boundary_condition_eta(values_H+values_h,values_hp, sd))*0.01+(values_H+values_h)*0.99 )**(4/3))

        values_u = values_u / (1 + sigma_q * dt / rho)
        values_v = values_v / (1 + sigma_q * dt / rho)

# -------------------------------------------------------------------------------------------------------------------
        self.boundary_condition_u(values_u,values_uu, sd)
        self.boundary_condition_v(values_v,values_vv, sd)
        eta1 = torch.maximum(k2,(values_H+values_h))
        eta2 = torch.maximum(k1,(values_H+values_h))
# -------------------------------------------------------------------------------------------------------------------
        b = beta * rho * (-self.xadv(self.boundary_condition_eta1(eta1,eta1_p, sd)) * values_u - self.yadv(self.boundary_condition_eta1(eta1,eta1_p, sd)) * values_v - eta1 * self.xadv(values_uu) - eta1 * self.yadv(values_vv) + self.PG_scalar(sd, self.boundary_condition_eta1(eta1,eta1_p, sd), eta1, values_u, values_v, k3, dx) - self.cmm(self.boundary_condition_dif_h(dif_values_h,dif_values_hh, sd)) / dt + source_h) / (dt * eta2)
        values_h_old = values_h.clone()
# -------------------------------------------------------------------------------------------------------------------
        for i in range(2):
            values_hh = values_hh - (-self.diff(self.boundary_condition_hh(values_hh,values_hhp, sd)) + beta * rho / (dt**2 * eta2) * values_hh) / (self.diag + beta * rho / (dt**2 * eta2)) + b / (self.diag + beta * rho / (dt**2 * eta2))
        values_h = values_h + values_hh
        dif_values_h = values_h - values_h_old
# -------------------------------------------------------------------------------------------------------------------
        values_u = values_u - self.xadv(self.boundary_condition_hh(values_hh,values_hhp, sd)) * dt / rho
        values_v = values_v - self.yadv(self.boundary_condition_hh(values_hh,values_hhp, sd)) * dt / rho
            
        return [values_u,values_v, values_uu, values_vv, values_H, values_h,values_hh, values_hp, values_hhp,dif_values_h, 
                dif_values_hh, source_h, k_x,k_y, sigma_q,eta1,eta2,eta1_p,    
                k1, k2, k3, dx,b,sd]



# compiled_model = [torch.compile(AI4SWE().to(device[0])), torch.compile(AI4SWE().to(device[1]))]
compiled_model = [AI4SWE().to(device[0]), AI4SWE().to(device[1])]
# compiled_model = AI4SWE()

def update_halos(sd):  # AMIN:: to optimise pass in as local vbls instead of global
    # device_no = sd.values_u.device # gpu number
    # --------------------------------------------------------------------------------- Bottom : send to 3 recevie for 0
    neig0 = sd.neig[0]
    if not isinstance(neig0, int):
        if neig0.values_u.device.index == sd.values_u.device.index:
            sd.halo_u[0]     = scale_it(neig0.values_u[0,0,0,:]                                , sd.halo_u[0])
            sd.halo_v[0]     = scale_it(neig0.values_v[0,0,0,:]                                , sd.halo_v[0])
            sd.halo_b_u[0]   = scale_it(neig0.b_u[0,0,0,:]                                     , sd.halo_b_u[0])
            sd.halo_b_v[0]   = scale_it(neig0.b_v[0,0,0,:]                                     , sd.halo_b_v[0])
            sd.halo_h[0]     = scale_it(neig0.values_h[0,0,0,:]                                , sd.halo_h[0])
            sd.halo_hh[0]    = scale_it(neig0.values_hh[0,0,0,:]                               , sd.halo_hh[0])
            sd.halo_dif_h[0] = scale_it(neig0.dif_values_h[0,0,0,:]                            , sd.halo_dif_h[0])
            sd.halo_eta[0]   = scale_it(neig0.values_H[0,0,0,:] + neig0.values_h[0,0,0,:]      , sd.halo_eta[0])
            sd.halo_eta1[0]  = scale_it(neig0.eta1[0,0,0,:]                                    , sd.halo_eta1[0])
        else:
            # Communication with the neighbour
            neig_rank = neig0.values_u.device.index
            SendTo = sd.values_u.device

            # sending data to neighbour
            dist.isend(tensor=sd.values_u[0,0,-1,:].contiguous()                               , dst=neig_rank ,tag = 140)
            dist.isend(tensor=sd.values_v[0,0,-1,:].contiguous()                               , dst=neig_rank ,tag = 141)
            dist.isend(tensor=sd.b_u[0,0,-1,:].contiguous()                                    , dst=neig_rank ,tag = 142)
            dist.isend(tensor=sd.b_v[0,0,-1,:].contiguous()                                    , dst=neig_rank ,tag = 143)
            dist.isend(tensor=sd.values_h[0,0,-1,:].contiguous()                               , dst=neig_rank ,tag = 144)
            dist.isend(tensor=sd.values_hh[0,0,-1,:].contiguous()                              , dst=neig_rank ,tag = 145)
            dist.isend(tensor=sd.dif_values_h[0,0,-1,:].contiguous()                           , dst=neig_rank ,tag = 146)
            dist.isend(tensor=(sd.values_H[0,0,-1,:] + sd.values_h[0,0,-1,:]).contiguous()     , dst=neig_rank ,tag = 147)
            dist.isend(tensor=sd.eta1[0,0,-1,:].contiguous()                                   , dst=neig_rank ,tag = 148)

            # receiving data from neighbour
            recv_410 = torch.zeros_like(neig0.values_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_410.contiguous(), src=neig_rank,tag=410)
            sd.halo_u[0] = scale_it(recv_410, sd.halo_u[0])

            recv_411 = torch.zeros_like(neig0.values_u[0,0,0,:]).to(SendTo)
            dist.recv(tensor=recv_411.contiguous(), src=neig_rank,tag=411)
            sd.halo_v[0] = scale_it(recv_411, sd.halo_v[0])

            recv_412 = torch.zeros_like(neig0.values_u[0,0,0,:]).to(SendTo)
            dist.recv(tensor=recv_412.contiguous(), src=neig_rank,tag=412)
            sd.halo_b_u[0] = scale_it(recv_412, sd.halo_b_u[0])

            recv_413 = torch.zeros_like(neig0.values_u[0,0,0,:]).to(SendTo)
            dist.recv(tensor=recv_413.contiguous(), src=neig_rank,tag=413)
            sd.halo_b_v[0] = scale_it(recv_413, sd.halo_b_v[0])

            recv_414 = torch.zeros_like(neig0.values_u[0,0,0,:]).to(SendTo)
            dist.recv(tensor=recv_414.contiguous(), src=neig_rank,tag=414)
            sd.halo_h[0] = scale_it(recv_414, sd.halo_h[0])

            recv_415 = torch.zeros_like(neig0.values_u[0,0,0,:]).to(SendTo)
            dist.recv(tensor=recv_415.contiguous(), src=neig_rank,tag=415)
            sd.halo_hh[0] = scale_it(recv_415, sd.halo_hh[0])

            recv_416 = torch.zeros_like(neig0.values_u[0,0,0,:]).to(SendTo)
            dist.recv(tensor=recv_416.contiguous(), src=neig_rank,tag=416)
            sd.halo_dif_h[0] = scale_it(recv_416, sd.halo_dif_h[0])

            recv_417 = torch.zeros_like(neig0.values_u[0,0,0,:]).to(SendTo)
            dist.recv(tensor=recv_417.contiguous(), src=neig_rank,tag=417)
            sd.halo_eta[0] = scale_it(recv_417, sd.halo_eta[0])

            recv_418 = torch.zeros_like(neig0.values_u[0,0,0,:]).to(SendTo)
            dist.recv(tensor=recv_418.contiguous(), src=neig_rank,tag=418)
            sd.halo_eta1[0] = scale_it(recv_418, sd.halo_eta1[0])

    # --------------------------------------------------------------------------------- left
    neig1 = sd.neig[1]
    if not isinstance(neig1, int):
        if neig1.values_u.device.index == sd.values_u.device.index:
            sd.halo_u[1]     = scale_it(neig1.values_u[0,0,:,-1]                           , sd.halo_u[1])
            sd.halo_v[1]     = scale_it(neig1.values_v[0,0,:,-1]                           , sd.halo_v[1])
            sd.halo_b_u[1]   = scale_it(neig1.b_u[0,0,:,-1]                                , sd.halo_b_u[1])
            sd.halo_b_v[1]   = scale_it(neig1.b_v[0,0,:,-1]                                , sd.halo_b_v[1])
            sd.halo_h[1]     = scale_it(neig1.values_h[0,0,:,-1]                           , sd.halo_h[1])
            sd.halo_hh[1]    = scale_it(neig1.values_hh[0,0,:,-1]                          , sd.halo_hh[1])
            sd.halo_dif_h[1] = scale_it(neig1.dif_values_h[0,0,:,-1]                       , sd.halo_dif_h[1])
            sd.halo_eta[1]   = scale_it(neig1.values_H[0,0,:,-1] + neig1.values_h[0,0,:,-1], sd.halo_eta[1])
            sd.halo_eta1[1]  = scale_it(neig1.eta1[0,0,:,-1]                               , sd.halo_eta1[1])
        else:
            # Communication with the neighbour
            neig_rank = neig1.values_u.device.index
            SendTo = sd.values_u.device

            # sending data to neighbour
            dist.isend(tensor=sd.values_u[0,0,:,0].contiguous()                            , dst=neig_rank ,tag = 230)
            dist.isend(tensor=sd.values_v[0,0,:,0].contiguous()                            , dst=neig_rank ,tag = 231)
            dist.isend(tensor=sd.b_u[0,0,:,0].contiguous()                                 , dst=neig_rank ,tag = 232)
            dist.isend(tensor=sd.b_v[0,0,:,0].contiguous()                                 , dst=neig_rank ,tag = 233)
            dist.isend(tensor=sd.values_h[0,0,:,0].contiguous()                            , dst=neig_rank ,tag = 234)
            dist.isend(tensor=sd.values_hh[0,0,:,0].contiguous()                           , dst=neig_rank ,tag = 235)
            dist.isend(tensor=sd.dif_values_h[0,0,:,0].contiguous()                        , dst=neig_rank ,tag = 236)
            dist.isend(tensor=(sd.values_H[0,0,:,0] + sd.values_h[0,0,:,0]).contiguous()   , dst=neig_rank ,tag = 237)
            dist.isend(tensor=sd.eta1[0,0,:,0].contiguous()                                , dst=neig_rank ,tag = 238)

            # receiving data from neighbour
            recv_320 = torch.zeros_like(neig1.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_320.contiguous(), src=neig_rank,tag=320)
            sd.halo_u[1] = scale_it(recv_320, sd.halo_u[1])

            recv_321 = torch.zeros_like(neig1.values_v[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_321.contiguous(), src=neig_rank,tag=321)
            sd.halo_v[1] = scale_it(recv_321, sd.halo_v[1])

            recv_322= torch.zeros_like(neig1.b_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_322.contiguous(), src=neig_rank,tag=322)
            sd.halo_b_u[1] = scale_it(recv_322, sd.halo_b_u[1])

            recv_323 = torch.zeros_like(neig1.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_323.contiguous(), src=neig_rank,tag=323)
            sd.halo_b_v[1] = scale_it(recv_323, sd.halo_b_v[1])

            recv_324 = torch.zeros_like(neig1.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_324.contiguous(), src=neig_rank,tag=324)
            sd.halo_h[1] = scale_it(recv_324, sd.halo_h[1])

            recv_325 = torch.zeros_like(neig1.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_325.contiguous(), src=neig_rank,tag=325)
            sd.halo_hh[1] = scale_it(recv_325, sd.halo_hh[1])

            recv_326 = torch.zeros_like(neig1.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_326.contiguous(), src=neig_rank,tag=326)
            sd.halo_dif_h[1] = scale_it(recv_326, sd.halo_dif_h[1])

            recv_327= torch.zeros_like(neig1.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_327.contiguous(), src=neig_rank,tag=327)
            sd.halo_eta[1] = scale_it(recv_327, sd.halo_eta[1])

            recv_328= torch.zeros_like(neig1.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_328.contiguous(), src=neig_rank,tag=328)
            sd.halo_eta1[1] = scale_it(recv_328, sd.halo_eta1[1])

    # --------------------------------------------------------------------------------- right
    neig2 = sd.neig[2]
    if not isinstance(neig2, int):
        if neig2.values_u.device.index == sd.values_u.device.index:
            sd.halo_u[2]     = scale_it(neig2.values_u[0,0,:,0]                            , sd.halo_u[2])
            sd.halo_v[2]     = scale_it(neig2.values_v[0,0,:,0]                            , sd.halo_v[2])
            sd.halo_b_u[2]   = scale_it(neig2.b_u[0,0,:,0]                                 , sd.halo_b_u[2])
            sd.halo_b_v[2]   = scale_it(neig2.b_v[0,0,:,0]                                 , sd.halo_b_v[2])
            sd.halo_h[2]     = scale_it(neig2.values_h[0,0,:,0]                            , sd.halo_h[2])
            sd.halo_hh[2]    = scale_it(neig2.values_hh[0,0,:,0]                           , sd.halo_hh[2])
            sd.halo_dif_h[2] = scale_it(neig2.dif_values_h[0,0,:,0]                        , sd.halo_dif_h[2])
            sd.halo_eta[2]   = scale_it(neig2.values_H[0,0,:,0] + neig2.values_h[0,0,:,0]  , sd.halo_eta[2])
            sd.halo_eta1[2]  = scale_it(neig2.eta1[0,0,:,0]                                , sd.halo_eta1[2])
        else:
            # Communication with the neighbour
            neig_rank = neig2.values_u.device.index
            SendTo = sd.values_u.device


            # receiving data from neighbour
            recv_230 = torch.zeros_like(neig2.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_230.contiguous(), src=neig_rank,tag=230)
            sd.halo_u[2] = scale_it(recv_230, sd.halo_u[2])

            recv_231 = torch.zeros_like(neig2.values_v[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_231.contiguous(), src=neig_rank,tag=231)
            sd.halo_v[2] = scale_it(recv_231, sd.halo_v[2])

            recv_232= torch.zeros_like(neig2.b_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_232.contiguous(), src=neig_rank,tag=232)
            sd.halo_b_u[2] = scale_it(recv_232, sd.halo_b_u[2])

            recv_233 = torch.zeros_like(neig2.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_233.contiguous(), src=neig_rank,tag=233)
            sd.halo_b_v[2] = scale_it(recv_233, sd.halo_b_v[2])

            recv_234 = torch.zeros_like(neig2.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_234.contiguous(), src=neig_rank,tag=234)
            sd.halo_h[2] = scale_it(recv_234, sd.halo_h[2])

            recv_235 = torch.zeros_like(neig2.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_235.contiguous(), src=neig_rank,tag=235)
            sd.halo_hh[2] = scale_it(recv_235, sd.halo_hh[2])

            recv_236 = torch.zeros_like(neig2.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_236.contiguous(), src=neig_rank,tag=236)
            sd.halo_dif_h[2] = scale_it(recv_236, sd.halo_dif_h[2])

            recv_237= torch.zeros_like(neig2.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_237.contiguous(), src=neig_rank,tag=237)
            sd.halo_eta[2] = scale_it(recv_237, sd.halo_eta[2])

            recv_238= torch.zeros_like(neig2.values_u[0,0,:,0].contiguous()).to(SendTo)
            dist.recv(tensor=recv_238.contiguous(), src=neig_rank,tag=238)
            sd.halo_eta1[2] = scale_it(recv_238, sd.halo_eta1[2])

            # sending data to neighbour
            dist.isend(tensor=sd.values_u[0,0,:,-1].contiguous()                           , dst=neig_rank ,tag = 320)
            dist.isend(tensor=sd.values_v[0,0,:,-1].contiguous()                           , dst=neig_rank ,tag = 321)
            dist.isend(tensor=sd.b_u[0,0,:,-1].contiguous()                                , dst=neig_rank ,tag = 322)
            dist.isend(tensor=sd.b_v[0,0,:,-1].contiguous()                                , dst=neig_rank ,tag = 323)
            dist.isend(tensor=sd.values_h[0,0,:,-1].contiguous()                           , dst=neig_rank ,tag = 324)
            dist.isend(tensor=sd.values_hh[0,0,:,-1].contiguous()                          , dst=neig_rank ,tag = 325)
            dist.isend(tensor=sd.dif_values_h[0,0,:,-1].contiguous()                       , dst=neig_rank ,tag = 326)
            dist.isend(tensor=(sd.values_H[0,0,:,-1] + sd.values_h[0,0,:,-1]).contiguous() , dst=neig_rank ,tag = 327)
            dist.isend(tensor=sd.eta1[0,0,:,-1].contiguous()                               , dst=neig_rank ,tag = 328)

    # --------------------------------------------------------------------------------- top: send to 0 receive for 3
    neig3 = sd.neig[3]
    if not isinstance(neig3, int):
        if neig3.values_u.device.index == sd.values_u.device.index:
            sd.halo_u[3]     = scale_it(neig3.values_u[0,0,-1,:]                               , sd.halo_u[3])
            sd.halo_v[3]     = scale_it(neig3.values_v[0,0,-1,:]                               , sd.halo_v[3])
            sd.halo_b_u[3]   = scale_it(neig3.b_u[0,0,-1,:]                                    , sd.halo_b_u[3])
            sd.halo_b_v[3]   = scale_it(neig3.b_v[0,0,-1,:]                                    , sd.halo_b_v[3])
            sd.halo_h[3]     = scale_it(neig3.values_h[0,0,-1,:]                               , sd.halo_h[3])
            sd.halo_hh[3]    = scale_it(neig3.values_hh[0,0,-1,:]                              , sd.halo_hh[3])
            sd.halo_dif_h[3] = scale_it(neig3.dif_values_h[0,0,-1,:]                           , sd.halo_dif_h[3])
            sd.halo_eta[3]   = scale_it(neig3.values_H[0,0,-1,:] + neig3.values_h[0,0,-1,:]    , sd.halo_eta[3])
            sd.halo_eta1[3]  = scale_it(neig3.eta1[0,0,-1,:]                                   , sd.halo_eta1[3])
        else:
            # Communication with the neighbour
            neig_rank = neig3.values_u.device.index
            SendTo = sd.values_u.device

            # receiving data from neighbour
            recv_140 = torch.zeros_like(neig3.values_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_140.contiguous(), src=neig_rank,tag=140)
            sd.halo_u[3] = scale_it(recv_140, sd.halo_u[3])

            recv_141 = torch.zeros_like(neig3.values_v[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_141.contiguous(), src=neig_rank,tag=141)
            sd.halo_v[3] = scale_it(recv_141, sd.halo_v[3])

            recv_142= torch.zeros_like(neig3.b_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_142.contiguous(), src=neig_rank,tag=142)
            sd.halo_b_u[3] = scale_it(recv_142, sd.halo_b_u[3])

            recv_143 = torch.zeros_like(neig3.values_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_143.contiguous(), src=neig_rank,tag=143)
            sd.halo_b_v[3] = scale_it(recv_143, sd.halo_b_v[3])

            recv_144 = torch.zeros_like(neig3.values_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_144.contiguous(), src=neig_rank,tag=144)
            sd.halo_h[3] = scale_it(recv_144, sd.halo_h[3])

            recv_145 = torch.zeros_like(neig3.values_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_145.contiguous(), src=neig_rank,tag=145)
            sd.halo_hh[3] = scale_it(recv_145, sd.halo_hh[3])

            recv_146 = torch.zeros_like(neig3.values_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_146.contiguous(), src=neig_rank,tag=146)
            sd.halo_dif_h[3] = scale_it(recv_146, sd.halo_dif_h[3])

            recv_147= torch.zeros_like(neig3.values_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_147.contiguous(), src=neig_rank,tag=147)
            sd.halo_eta[3] = scale_it(recv_147, sd.halo_eta[3])

            recv_148= torch.zeros_like(neig3.values_u[0,0,0,:].contiguous()).to(SendTo)
            dist.recv(tensor=recv_148.contiguous(), src=neig_rank,tag=148)
            sd.halo_eta1[3] = scale_it(recv_148, sd.halo_eta1[3])
            
            # sending data to neighbour
            dist.send(tensor=sd.values_u[0,0,0,:].contiguous()                                 , dst=neig_rank ,tag = 410)
            dist.send(tensor=sd.values_v[0,0,0,:].contiguous()                                 , dst=neig_rank ,tag = 411)
            dist.send(tensor=sd.b_u[0,0,0,:].contiguous()                                      , dst=neig_rank ,tag = 412)
            dist.send(tensor=sd.b_v[0,0,0,:].contiguous()                                      , dst=neig_rank ,tag = 413)
            dist.send(tensor=sd.values_h[0,0,0,:].contiguous()                                 , dst=neig_rank ,tag = 414)
            dist.send(tensor=sd.values_hh[0,0,0,:].contiguous()                                , dst=neig_rank ,tag = 415)
            dist.send(tensor=sd.dif_values_h[0,0,0,:].contiguous()                             , dst=neig_rank ,tag = 416)
            dist.send(tensor=(sd.values_H[0,0,0,:] + sd.values_h[0,0,0,:]).contiguous()        , dst=neig_rank ,tag = 417)
            dist.send(tensor=sd.eta1[0,0,0,:].contiguous()                                     , dst=neig_rank ,tag = 418)
    return


real_time = 0
def run_on_device(args):
    with torch.cuda.amp.autocast(dtype=torch.float16):
        model, sd_list, cuda_indx = args
        torch.cuda.set_device(cuda_indx)
        init_process(cuda_indx)
        global real_time
        start = time.time()
        with torch.no_grad():
            for istep in range(3601):
                img = int(real_time/900)
                for sd in sd_list:
                    get_source(sd, real_time)
                    update_halos(sd)
                    for t in range(2):
                        [sd.values_u,      sd.values_v, sd.values_uu, sd.values_vv, 
                        sd.values_H,      sd.values_h, sd.values_hh, sd.values_hp, sd.values_hhp, sd.dif_values_h, 
                        sd.dif_values_hh, sd.source_h, sd.k_x,       sd.k_y,       sd.sigma_q,    sd.eta1,      sd.eta2,      sd.eta1_p,    
                        sd.k1,            sd.k2,       sd.k3,        sd.dx,        sd.b,          sd]  = model( sd.values_u,      sd.values_v, sd.values_uu, sd.values_vv, 
                        sd.values_H,      sd.values_h, sd.values_hh, sd.values_hp, sd.values_hhp, sd.dif_values_h, 
                        sd.dif_values_hh, sd.source_h, sd.k_x,       sd.k_y,       sd.sigma_q,    sd.eta1,      sd.eta2,       sd.eta1_p,    
                        sd.k1,            sd.k2,       sd.k3,        sd.dx,        sd.b,          sd,           dt, rho )
                                
                    real_time = real_time + dt

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------
                    if (istep*dt) % 900 <= 0.001:
                        img = int(real_time/900)
                        print(f'Time step:, {istep}, sd_rank:{sd.rank}, img = {int(real_time/900)} , time in seconds = {real_time:.0f}', 'wall clock:', (time.time()-start)/60)

                        np.save(f'/home/an619/Desktop/git/AI/RMS/semi/2D/Linear_results/eta_{sd.rank}_{istep}', (sd.values_H[0,0,:,:]+sd.values_h[0,0,:,:]).cpu())
                        plt.imshow((sd.values_h+sd.values_H)[0,0,:,:].cpu().detach().numpy())
                        plt.savefig(f'/home/an619/Desktop/git/AI/RMS/semi/2D/Linear_results/{img}_{sd.rank}.png', dpi=200, bbox_inches='tight')

            end = time.time()
            print('time',(end-start),istep)

sd_list[0].to('cuda:0')
sd_list[1].to('cuda:1')
sd_list[2].to('cuda:0')
sd_list[3].to('cuda:1')

sd_list0 = [sd_list[0], sd_list[2]]
sd_list1 = [sd_list[1], sd_list[3]]

# if __name__ == "__main__":
#     set_start_method('spawn')

#     p0 = Process(target=run_on_device, args=(compiled_model[0], sd_list0, 0))
#     p1 = Process(target=run_on_device, args=(compiled_model[1], sd_list1, 1))

#     p0.start()
#     print('p0 started')
#     p1.start()
#     print('p1 started')

#     p0.join()
#     print('p0 joined')
#     p1.join()
#     print('p1 joined')

from multiprocessing import Pool, set_start_method
if __name__ == "__main__":
    set_start_method('spawn')

    args = [(compiled_model[0], sd_list0, 0), (compiled_model[1], sd_list1, 1)]

    with Pool(2) as p:
        print('Pool started')
        p.map(run_on_device, args)
        print('Pool joined')