import tensorflow as tf
import torch
from tensorflow import keras
import numpy as np
import vtk
import matplotlib.pyplot as plt

class subdomain_2D:
    # this is for quadratic elemetns
    def __init__(self, nx, ny, dx, dy, device, dt, eplsion_eta, rank):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.device = device
        self.dt = dt
        self.eplsion_eta = eplsion_eta
        self.diag = None

        self.w = None
        
        self.rank = rank
        self.neig = ['side' for _ in range(4)]
        self.corner_neig = None
        self.corner_node_neig = None

        shape = (1, 1, ny, nx)
        input_shape = (1, 1, ny, nx)
        self.values_h = torch.zeros(input_shape, device=device)
        self.values_H = torch.zeros(input_shape, device=device)
        self.values_u = torch.zeros(input_shape, device=device)
        self.values_v = torch.zeros(input_shape, device=device)
        self.a_u = torch.zeros(input_shape, device=device)
        self.a_v = torch.zeros(input_shape, device=device)
        self.b_u = torch.zeros(input_shape, device=device)
        self.b_v = torch.zeros(input_shape, device=device)
        self.eta1 = torch.zeros(input_shape, device=device)
        self.eta2 = torch.zeros(input_shape, device=device)
        self.values_hh = torch.zeros(input_shape, device=device)
        self.dif_values_h = torch.zeros(input_shape, device=device)
        self.values_h_old = torch.zeros(input_shape, device=device)
        self.sigma_q = torch.zeros(input_shape, device=device)
        self.k_u = torch.zeros(input_shape, device=device)
        self.k_v = torch.zeros(input_shape, device=device)
        self.k_x = torch.zeros(input_shape, device=device)
        self.k_y = torch.zeros(input_shape, device=device)
        self.b = torch.zeros(input_shape, device=device)
        self.source_h = torch.zeros(input_shape, device=device)
        self.b = torch.zeros(input_shape, device=device)
        self.H = torch.zeros(input_shape, device=device)

        # stablisation factor
        self.k1 = torch.ones(input_shape, device=device)*eplsion_eta
        self.k2 = torch.zeros(input_shape, device=device)
        self.k3 = torch.ones(input_shape, device=device)*dx**2*0.1/dt
        self.kmax = None
        self.m_i = None
        self.pg_cst = None

        # Padding
        input_shape_pd = (1, 1, ny + 4, nx + 4)
        self.values_uu = torch.zeros(input_shape_pd, device=device)
        self.values_vv = torch.zeros(input_shape_pd, device=device)
        self.b_uu = torch.zeros(input_shape_pd, device=device)
        self.b_vv = torch.zeros(input_shape_pd, device=device)
        self.eta1_p = torch.zeros(input_shape_pd, device=device)
        self.dif_values_hh = torch.zeros(input_shape_pd, device=device)
        self.values_hhp = torch.zeros(input_shape_pd, device=device)
        self.values_hp = torch.zeros(input_shape_pd, device=device)
        self.values_Hp = torch.zeros(input_shape_pd, device=device)
        self.k_uu = torch.zeros(input_shape_pd, device=device)
        self.k_vv = torch.zeros(input_shape_pd, device=device)
        self.pad_H = torch.zeros(input_shape_pd, device=device)

        self.values_hhp_L = torch.zeros((1, 1, ny + 2, nx + 2), device=device)
        
        # Halos
        self.halo_u     = torch.zeros((4, ny+4, nx+4), device=device)
        self.halo_v     = torch.zeros((4, ny+4, nx+4), device=device)
        self.halo_h     = torch.zeros((4, ny+4, nx+4), device=device)
        self.halo_hh    = torch.zeros((4, ny+4, nx+4), device=device)
        self.halo_dif_h = torch.zeros((4, ny+4, nx+4), device=device)
        self.halo_eta   = torch.zeros((4, ny+4, nx+4), device=device)
        self.halo_eta1  = torch.zeros((4, ny+4, nx+4), device=device)
        
        # CNN models
        self.dif = None
        self.xadv = None
        self.yadv = None
        self.CNN3D_Su = None
        self.CNN3D_Sv = None
        self.CNN3D_pu = None
        self.CNN3D_pv = None
        self.CNN3D_A_padd = None
        self.CNN3D_A = None

        # if you want tensorflow arrays use thses lines instead
        # self.values_u = tf.zeros(shape)
        # self.values_v = tf.zeros(shape)
        # self.values_w = tf.zeros(shape)
        # self.values_p = tf.zeros(shape)
        # self.b = tf.zeros(shape)
        # self.halo_u = tf.zeros((6, ny, nx))
        # self.halo_v = tf.zeros((6, ny, nx))
        # self.halo_w = tf.zeros((6, ny, nx))
        # self.halo_p = tf.zeros((6, ny, nx))



def init_grid_levels(sd):
    sd.w = []
    nx = sd.nx
    for i in range(sd.nlevel-1):
        sd.w.append(np.zeros((1,nx, nx, nx, 1)))
        nx //= 2


def plot_subdomains(no_domains, no_domains_x, no_domains_y, sd):
    '''
    This function plots the subdomains in a grid.   
    '''
    # Calculate the size of each subplot
    subplot_size_x = 5  # You can adjust this value as needed
    subplot_size_y = 5  # You can adjust this value as needed

    # Calculate the total figure size
    fig_size_x = subplot_size_x * no_domains_x
    fig_size_y = subplot_size_y * no_domains_y
    
    fig, axs = plt.subplots(no_domains_y, no_domains_x, figsize=(fig_size_x, fig_size_y))

    for i in range(no_domains):
        row = no_domains_y - 1 - i // no_domains_x
        col = i % no_domains_x
        axs[row, col].imshow(sd[i].values_h[0,0,:,:].cpu()+sd[i].values_H[0,0,:,:].cpu())

    plt.show()


def save_data(sd,n_out,itime, w):
    # if w means if it is 3D which can be True or False
    if itime % n_out == 0:  
        np.save("result_bluff_body/result_3d_BC/u"+str(itime), arr = sd.values_u[0,:,:,:,0])
        np.save("result_bluff_body/result_3d_BC/v"+str(itime), arr = sd.values_v[0,:,:,:,0])
        np.save("result_bluff_body/result_3d_BC/p"+str(itime), arr = sd.values_p[0,:,:,:,0])
        if w:
            np.save("result_bluff_body/result_3d_BC/w"+str(itime), arr = sd.values_w[0,:,:,:,0])

# ----------------------------------------------------------------------- stencils for 3D:
# ----------------------------------- FDM -------------------------------
# # Diffusion
# diff1 = [[0.0, 0.0, 0.0],
#         [ 0.0, 1.0, 0.0],
#         [ 0.0, 0.0, 0.0]]

# diff2 = [[0.0,  1.0, 0.0],
#         [ 1.0, -6.0, 1.0],
#         [ 0.0,  1.0, 0.0]]

# diff3 = [[0.0, 0.0, 0.0],
#         [ 0.0, 1.0, 0.0],
#         [ 0.0, 0.0, 0.0]]

# # Advection:
# adv_x1 = [[0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0]]

# adv_x2 = [[0.0, 0.0,  0.0],
#           [0.5, 0.0, -0.5],
#           [0.0, 0.0,  0.0]]

# adv_x3 = [[0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0]]


# adv_y1 = [[0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0]]

# adv_y2 = [[0.0, -0.5, 0.0],
#           [0.0,  0.0, 0.0],
#           [0.0,  0.5, 0.0]]

# adv_y3 = [[0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0]]


# adv_z1 = [[0.0, 0.0, 0.0],
#           [0.0, 0.5, 0.0],
#           [0.0, 0.0, 0.0]]

# adv_z2 = [[0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0]]

# adv_z3 = [[0.0,  0.0, 0.0],
#           [0.0, -0.5, 0.0],
#           [0.0,  0.0, 0.0]]

# ----------------------------------- FEM -------------------------------
# Diffusion
diff1 = [[2/26, 3/26,  2/26],
        [ 3/26, 6/26,  3/26],
        [ 2/26, 3/26,  2/26]]

diff2 = [[3/26,  6/26,  3/26],
        [ 6/26, -88/26, 6/26],
        [ 3/26,  6/26,  3/26]]

diff3 = [[2/26, 3/26,  2/26],
        [ 3/26, 6/26,  3/26],
        [ 2/26, 3/26,  2/26]]

# Advection:
adv_x1 = [[0.014,  0.0, -0.014 ],
          [ 0.056, 0.0, -0.056],
          [ 0.014, 0.0, -0.014]]

adv_x2 = [[0.056,  0.0, -0.056 ],
          [ 0.22,  0.0, -0.22 ],
          [ 0.056, 0.0, -0.056]]

adv_x3 = [[0.014,  0.0, -0.014 ],
          [ 0.056, 0.0, -0.056],
          [ 0.014, 0.0, -0.014]]


adv_y1 = [[-0.014, -0.056, -0.014],
          [ 0.0,    0.0,    0.0  ],
          [ 0.014,  0.056,  0.014]]

adv_y2 = [[-0.056, -0.22, -0.056],
          [ 0.0,    0.0,   0.0  ],
          [ 0.056,  0.22,  0.056]]

adv_y3 = [[-0.014, -0.056, -0.014],
          [ 0.0,    0.0,    0.0  ],
          [ 0.014,  0.056,  0.014]]


adv_z1 = [[0.014, 0.056, 0.014],
          [0.056, 0.22,  0.056],
          [0.014, 0.056, 0.014]]

adv_z2 = [[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]]

adv_z3 = [[-0.014, -0.056, -0.014],
          [-0.056, -0.22,  -0.056],
          [-0.014, -0.056, -0.014]]


def set_face_neigs(sd, width, height):
    '''
    it defines all neighbours for each subdomains. 
    side: when it is on the domain bc
    inlet: when it is at the inlet
    outlet: when it is at the outlet
    sd nfo: when there is another sd as a neig
    sd: is a list of all subdomains
    width: number of domains in x direction
    height: number of domains in y direction
    
    # Example: Accessing the neigs of sd[3]
    ele = 1
    for iface in range(6):
        if not isinstance(sd[ele].neig[iface], str):
            print(sd[ele].neig[iface].rank)
        else:
            print(sd[ele].neig[iface])
    '''
    for i in range(len(sd)):
        rank = sd[i].rank

        # Check if the current domain is on the bottom row
        if rank - width < 0:
            sd[i].neig[0] = -1 # 'bottom'
        else:
            sd[i].neig[0] = sd[rank - width]

        # Check if the current domain is on the leftmost column
        if rank % width == 0:
            sd[i].neig[1] = -1 # 'left'
        else:
            sd[i].neig[1] = sd[rank - 1]

        # Check if the current domain is on the rightmost column
        if (rank+1) % width == 0:
            sd[i].neig[2] = -1 # 'right'
        else:
            sd[i].neig[2] = sd[rank + 1]

        # Check if the current domain is on the top row
        if rank >= width*(height-1):
            sd[i].neig[3] = -1 # 'top'
        else:
            sd[i].neig[3] = sd[rank + width]


def set_corner_neighbors(sd, no_domains):
    '''
    corner node numbering starts from bottom left and is row major
    2 ------- 3
      |      |
    0 ------- 1
    '''

    for ele in range(no_domains):
        sd[ele].corner_neig = [-1,-1,-1,-1]
        # ------------------------------------------------------------------------ bottom left & right
        neig0 = sd[ele].neig[0]
        if neig0 != -1 :
            if neig0.neig[1] != -1:
                sd[ele].corner_neig[0] = neig0.neig[1]
            if neig0.neig[2] != -1:
                sd[ele].corner_neig[1] = neig0.neig[2]
        # ------------------------------------------------------------------------ top left & right
        neig3 = sd[ele].neig[3]
        if neig3 != -1 :
            if neig3.neig[1] != -1:
                sd[ele].corner_neig[2] = neig3.neig[1]
            if neig3.neig[2] != -1:
                sd[ele].corner_neig[3] = neig3.neig[2]
    return

    




# def set_corner_neighbors(sd, no_elements, num_rows, num_cols):
#     '''
#     It works only if num_ele in z-dir is 1. otherwise you should modify it
#     it find corner neig of each side. to find corner neig of each corner node, you should use another function
#     '''
    
#     def get_inner_corner_neighbor(corner_idx):
#         if corner_idx == 9:  # Bottom left
#             return sd[ele - num_cols - 1]
#         elif corner_idx == 10:  # Bottom right
#             return sd[ele - num_cols + 1]
#         elif corner_idx == 5:  # Top left
#             return sd[ele + num_cols - 1]
#         elif corner_idx == 6:  # Top right
#             return sd[ele + num_cols + 1]
#     for ele in range (no_elements):
#         row = ele // num_cols
#         col = ele % num_cols
        
#         sd[ele].corner_neig = ['No'] * 4
        
#         if sd[ele].neig[0] != -1 :# 'bottom':
#             if col ==0:
#                 sd[ele].corner_neig[1] = sd[sd[ele].neig[0].rank+1]
#             elif col == num_cols-1:
#                 sd[ele].corner_neig[0] = sd[sd[ele].neig[0].rank-1]
#             else:
#                 sd[ele].corner_neig[0] = sd[sd[ele].neig[0].rank-1]
#                 sd[ele].corner_neig[1] = sd[sd[ele].neig[0].rank+1]

#         if sd[ele].neig[3] != -1 :# 'top':
#             if col ==0:
#                 sd[ele].corner_neig[3] = sd[sd[ele].neig[3].rank+1]
#             elif col == num_cols-1:
#                 sd[ele].corner_neig[2] = sd[sd[ele].neig[3].rank-1]
#             else:
#                 sd[ele].corner_neig[2] = sd[sd[ele].neig[3].rank-1]
#                 sd[ele].corner_neig[3] = sd[sd[ele].neig[3].rank+1]
            



def set_corner_node_neig(sd, no_domains):
    # for 3D case only
    # corner_neig_points = coner points of cornet arrays which in tthis case are empty. you must defin them for cases for example 3X3X3 subdomains
    # here becasue we have only 3X3X1 we do not need to defin it
    
    for i in range(no_domains):
        sd[i].corner_node_neig = ['No'] * 8


def get_3D_libraries(sd, u_x, dt,Re):
    nx = sd.nx ; ny = sd.ny; nz = sd.nz
    dx = sd.dx ; nlevel = sd.nlevel
    # --------------------------------------------------------------------------------- bias term
    bias_initializer = tf.keras.initializers.constant(np.zeros((1,)))
    # ------------------------------------------------------------------------- Libraries for solving momentum equation
    w1 = np.zeros([1,3,3,3,1])
    w2 = np.zeros([1,3,3,3,1])
    w3 = np.zeros([1,3,3,3,1])
    w4 = np.zeros([1,3,3,3,1])
    w6 = np.zeros([1,3,3,3,1])
    w7 = np.zeros([1,3,3,3,1])
    w8 = np.zeros([1,3,3,3,1])

    w1[0,0,:,:,0] = np.array(diff1)#*dt*Re/dx**2
    w1[0,1,:,:,0] = np.array(diff2)#*dt*Re/dx**2 
    w1[0,2,:,:,0] = np.array(diff3)#*dt*Re/dx**2 

    w2[0,0,:,:,0] = np.array(adv_x1)*u_x*dt/dx*(-1)
    w2[0,1,:,:,0] = np.array(adv_x2)*u_x*dt/dx*(-1) 
    w2[0,2,:,:,0] = np.array(adv_x3)*u_x*dt/dx*(-1)

    w3[0,0,:,:,0] = np.array(adv_y1)*u_x*dt/dx*(-1)
    w3[0,1,:,:,0] = np.array(adv_y2)*u_x*dt/dx*(-1)
    w3[0,2,:,:,0] = np.array(adv_y3)*u_x*dt/dx*(-1)

    w4[0,0,:,:,0] = np.array(adv_z1)*u_x*dt/dx*(-1) 
    w4[0,1,:,:,0] = np.array(adv_z2)*u_x*dt/dx*(-1)
    w4[0,2,:,:,0] = np.array(adv_z3)*u_x*dt/dx*(-1)

    w6[0,0,:,:,0] = np.array(adv_x1)/dt/dx*(-1)
    w6[0,1,:,:,0] = np.array(adv_x2)/dt/dx*(-1) 
    w6[0,2,:,:,0] = np.array(adv_x3)/dt/dx*(-1) 
    w7[0,0,:,:,0] = np.array(adv_y1)/dt/dx*(-1) 
    w7[0,1,:,:,0] = np.array(adv_y2)/dt/dx*(-1) 
    w7[0,2,:,:,0] = np.array(adv_y3)/dt/dx*(-1) 
    w8[0,0,:,:,0] = np.array(adv_z1)/dt/dx*(-1) 
    w8[0,1,:,:,0] = np.array(adv_z2)/dt/dx*(-1) 
    w8[0,2,:,:,0] = np.array(adv_z3)/dt/dx*(-1) 



    kernel_initializer_1 = tf.keras.initializers.constant(w1)
    kernel_initializer_2 = tf.keras.initializers.constant(w2)
    kernel_initializer_3 = tf.keras.initializers.constant(w3)
    kernel_initializer_4 = tf.keras.initializers.constant(w4)

    kernel_initializer_6 = tf.keras.initializers.constant(w6)
    kernel_initializer_7 = tf.keras.initializers.constant(w7)
    kernel_initializer_8 = tf.keras.initializers.constant(w8)


    input_shape = (nx+2, ny+2, nz+2, 1)
    sd.CNN3D_dif = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_1,
                                    bias_initializer=bias_initializer),
    ])

    sd.CNN3D_xadv = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_2,
                                    bias_initializer=bias_initializer),
    ])

    sd.CNN3D_yadv = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_3,
                                    bias_initializer=bias_initializer),
    ])

    sd.CNN3D_zadv = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_4,
                                    bias_initializer=bias_initializer),
    ])

    sd.CNN3D_Su = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_6,
                                    bias_initializer=bias_initializer),
    ])

    sd.CNN3D_Sv = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_7,
                                    bias_initializer=bias_initializer),
    ])

    sd.CNN3D_Sw = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_8,
                                    bias_initializer=bias_initializer),
    ])


    sd.CNN3D_pu = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_2,
                                    bias_initializer=bias_initializer),
    ])

    sd.CNN3D_pv = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_3,
                                    bias_initializer=bias_initializer),
    ])

    sd.CNN3D_pw = keras.models.Sequential([
            keras.layers.InputLayer(input_shape= input_shape),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                    kernel_initializer=kernel_initializer_4,
                                    bias_initializer=bias_initializer),
    ])


    # --------------------------------------------------------------------------------- Libraries for solving the Poisson equation
    # A matrix for Jacobi
    sd.CNN3D_A = []
    sd.CNN3D_A_padd = []

    sd.w_A = np.zeros([1,3,3,3,1])
    sd.w_A[0,0,:,:,0] = -np.array(diff1)/dx**2
    sd.w_A[0,1,:,:,0] = -np.array(diff2)/dx**2 
    sd.w_A[0,2,:,:,0] = -np.array(diff3)/dx**2

    kernel_initializer_A = tf.keras.initializers.constant(sd.w_A)

    for i in range(nlevel):
        CNN3D_A_i = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(int(nz*0.5**(nlevel-1-i))+2,int(ny*0.5**(nlevel-1-i))+2, int(nx*0.5**(nlevel-1-i))+2, 1)),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         
                                kernel_initializer=kernel_initializer_A,
                                bias_initializer=bias_initializer)
        ])
        sd.CNN3D_A_padd.append(CNN3D_A_i)  # Append the model to the list 
    sd.CNN3D_A_padd.reverse()


    for i in range(nlevel):
        CNN3D_A_i = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(int(nz*0.5**(nlevel-1-i)),int(ny*0.5**(nlevel-1-i)), int(nx*0.5**(nlevel-1-i)), 1)),
            tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         
                                kernel_initializer=kernel_initializer_A,
                                bias_initializer=bias_initializer)
        ])
        sd.CNN3D_A.append(CNN3D_A_i)  # Append the model to the list 
    sd.CNN3D_A.reverse()


    # ----------------------------------------------------------------------------------------- restrictor
    w9 = np.zeros([1,2,2,2,1])

    w9[0,:,:,:,0] = 0.125
    kernel_initializer_9 = tf.keras.initializers.constant(w9)
    sd.CNN3D_res = []

    w1 = np.zeros([1,2,2,2,1])
    w1[0,:,:,:,0] = 0.125
    kernel_initializer_1 = tf.keras.initializers.constant(w1)

    for i in range(nlevel-1):
        CNN3D_res_i = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(int(nz*0.5**(nlevel-2-i)), int(ny*0.5**(nlevel-2-i)), int(nx*0.5**(nlevel-2-i)), 1)),
            tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                    kernel_initializer=kernel_initializer_1,
                                    bias_initializer=bias_initializer),   
        ])   

        sd.CNN3D_res.append(CNN3D_res_i)


    # ------------------------------------------------------------------------------------ prolongator
    sd.CNN3D_prol = []

    for i in range(nlevel-1):
        CNN3D_prol_i = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(1*2**i,1*2**i, 1*2**i, 1)),
            tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
        ])
        sd.CNN3D_prol.append(CNN3D_prol_i)

    return



def describe(sd):
    print(f'------------------------------------- description of sd{sd.rank} ---------------------------------------------------')
    print('nx,ny,nz:         ', sd.nx,sd.ny,sd.nz)
    print('dx, dy, dz:', sd.dx, sd.dy, sd.dz)
    print('nlevel:           ', sd.nlevel)
    print('sd.w shape:       ', np.shape(sd.values_w))
    print('neig:             ', sd.neig)
    print('corner neig:      ', sd.corner_neig)
    print('corner node neig: ', sd.corner_node_neig)
    print(f'----------------------------------------------------------------------------------------------------------------------------')


def physical_halos_vel(sd, values_u,values_v,values_w):
    for iface in range(6):
        if sd.neig[iface] == 'inlet' :
            sd.halo_u[2, :, :] = 1#ub                                                # left
            sd.halo_v[2, :, :] = 0.0    
            sd.halo_w[2, :, :] = 0.0
        
        elif sd.neig[iface] == 'outlet':
            sd.halo_u[3, :, :] = 1#ub                                                # right
            sd.halo_v[3, :, :] = 0.0    
            sd.halo_w[3, :, :] = 0.0
            
        elif sd.neig[iface] == 'side':
            if iface == 0:
                sd.halo_u[0, 1:-1, 1:-1] = values_u[0,1,:,:,0]                  # front
                sd.halo_v[0, :, :] = 0.0
                sd.halo_w[0, :, :] = 0.0
            elif iface == 1:
                sd.halo_u[1, 1:-1, 1:-1] = values_u[0,-2,:,:,0]                 # back
                sd.halo_v[1, :, :] = 0.0    
                sd.halo_w[1, :, :] = 0.0
            elif iface == 4:
                sd.halo_u[4, 1:-1, 1:-1] = values_u[0,:,1,:,0]                  # top
                sd.halo_v[4, :, :] = 0.0
                sd.halo_w[4, :, :] = 0.0
            elif iface == 5:
                sd.halo_u[5, 1:-1, 1:-1] = values_u[0,:,-2,:,0]                 # bottom
                sd.halo_v[5, :, :] = 0.0   
                sd.halo_w[5, :, :] = 0.0

                
def read_physical_halos_p(sd):   
    for iface in range(6):
        if sd.neig[iface] == 'outlet' :
            sd.halo_p[3, :, :]  = 0.0 
        elif sd.neig[iface] == 'inlet' or sd.neig[iface] == 'side':
            if iface == 0:
                sd.halo_p[0, 1:-1, 1:-1]  = sd.values_p[0, 1, :, :,0]              # front
            elif iface == 1:
                sd.halo_p[1, 1:-1, 1:-1]  = sd.values_p[0,-2, :, :,0]              # back
            elif iface == 2:
                sd.halo_p[2, 1:-1, 1:-1]  = sd.values_p[0, :, :, 1,0]              # left 
            elif iface == 4:
                sd.halo_p[4, 1:-1, 1:-1]  = sd.values_p[0, :, 1, :,0]              # top
            elif iface == 5:
                sd.halo_p[5, 1:-1, 1:-1]  = sd.values_p[0, :,-2, :,0]              # bottom
                

                
def update_inner_halo_vel(sd):
    for iface in range(6):
        if not isinstance(sd.neig[iface], str):
            if iface == 0:
                pass # complete it later for other case
            elif iface == 1:
                pass # complete it later for other case
            elif iface == 2:
                sd.halo_u[2, 1:-1, 1:-1] = scale_faces(sd.values_u[0,:,:,1,0], sd.neig[iface].values_u[0,:,:,-1,0])                # left 
                sd.halo_v[2, 1:-1, 1:-1] = scale_faces(sd.values_v[0,:,:,1,0], sd.neig[iface].values_v[0,:,:,-1,0])
                sd.halo_w[2, 1:-1, 1:-1] = scale_faces(sd.values_w[0,:,:,1,0], sd.neig[iface].values_w[0,:,:,-1,0])
            elif iface == 3:
                sd.halo_u[3, 1:-1, 1:-1] = scale_faces(sd.values_u[0,:,:,-2,0], sd.neig[iface].values_u[0,:,:,0,0])                  # right 
                sd.halo_v[3, 1:-1, 1:-1] = scale_faces(sd.values_v[0,:,:,-2,0], sd.neig[iface].values_v[0,:,:,0,0])
                sd.halo_w[3, 1:-1, 1:-1] = scale_faces(sd.values_w[0,:,:,-2,0], sd.neig[iface].values_w[0,:,:,0,0])
            elif iface == 4:
                sd.halo_u[4, 1:-1, 1:-1] = scale_faces(sd.values_u[0,:,1,:,0], sd.neig[iface].values_u[0,:,-1,:,0])                  # top
                sd.halo_v[4, 1:-1, 1:-1] = scale_faces(sd.values_v[0,:,1,:,0], sd.neig[iface].values_v[0,:,-1,:,0])
                sd.halo_w[4, 1:-1, 1:-1] = scale_faces(sd.values_w[0,:,1,:,0], sd.neig[iface].values_w[0,:,-1,:,0])
            elif iface == 5:
                sd.halo_u[5, 1:-1, 1:-1] = scale_faces(sd.values_u[0,:,-2,:,0], sd.neig[iface].values_u[0,:,0,:,0])                  # bottom
                sd.halo_v[5, 1:-1, 1:-1] = scale_faces(sd.values_v[0,:,-2,:,0], sd.neig[iface].values_v[0,:,0,:,0])
                sd.halo_w[5, 1:-1, 1:-1] = scale_faces(sd.values_w[0,:,-2,:,0], sd.neig[iface].values_w[0,:,0,:,0])
            
            
            
def update_inner_halo_p(sd):
    for iface in range(6):
        if not isinstance(sd.neig[iface], str):
            if iface == 0:
                pass # complete it later for other case
            elif iface == 1:
                pass # complete it later for other case
            elif iface == 2:
                sd.halo_p[2, 1:-1, 1:-1] = scale_faces(sd.values_p[0,:,:,1,0], sd.neig[iface].values_p[0,:,:,-1,0])                 # left 
            elif iface == 3:
                sd.halo_p[3, 1:-1, 1:-1] = scale_faces(sd.values_p[0,:,:,-2,0], sd.neig[iface].values_p[0,:,:,0,0])                  # right
            elif iface == 4:
                sd.halo_p[4, 1:-1, 1:-1] = scale_faces(sd.values_p[0,:,1,:,0], sd.neig[iface].values_p[0,:,-1,:,0])                 # top
            elif iface == 5:
                sd.halo_p[5, 1:-1, 1:-1] = scale_faces(sd.values_p[0,:,-2,:,0], sd.neig[iface].values_p[0,:,0,:,0])                  # top
    

def read_halos(sd):
    for i in range(len(sd)):
        sd[i].values_u[0, 0,:,:,0] = sd[i].halo_u[0,1:-1,1:-1]
        sd[i].values_u[0,-1,:,:,0] = sd[i].halo_u[1,1:-1,1:-1]
        sd[i].values_u[0,:, 0,:,0] = sd[i].halo_u[4,1:-1,1:-1]
        sd[i].values_u[0,:,-1,:,0] = sd[i].halo_u[5,1:-1,1:-1]
        sd[i].values_u[0,:,:, 0,0] = sd[i].halo_u[2,1:-1,1:-1]
        sd[i].values_u[0,:,:,-1,0] = sd[i].halo_u[3,1:-1,1:-1]
        
        sd[i].values_v[0, 0,:,:,0] = sd[i].halo_v[0,1:-1,1:-1]
        sd[i].values_v[0,-1,:,:,0] = sd[i].halo_v[1,1:-1,1:-1]
        sd[i].values_v[0,:, 0,:,0] = sd[i].halo_v[4,1:-1,1:-1]
        sd[i].values_v[0,:,-1,:,0] = sd[i].halo_v[5,1:-1,1:-1]
        sd[i].values_v[0,:,:, 0,0] = sd[i].halo_v[2,1:-1,1:-1]
        sd[i].values_v[0,:,:,-1,0] = sd[i].halo_v[3,1:-1,1:-1]
        
        sd[i].values_w[0, 0,:,:,0] = sd[i].halo_w[0,1:-1,1:-1]
        sd[i].values_w[0,-1,:,:,0] = sd[i].halo_w[1,1:-1,1:-1]
        sd[i].values_w[0,:, 0,:,0] = sd[i].halo_w[4,1:-1,1:-1]
        sd[i].values_w[0,:,-1,:,0] = sd[i].halo_w[5,1:-1,1:-1]
        sd[i].values_w[0,:,:, 0,0] = sd[i].halo_w[2,1:-1,1:-1]
        sd[i].values_w[0,:,:,-1,0] = sd[i].halo_w[3,1:-1,1:-1]
        
        sd[i].values_p[0, 0,:,:,0] = sd[i].halo_p[0,1:-1,1:-1]
        sd[i].values_p[0,-1,:,:,0] = sd[i].halo_p[1,1:-1,1:-1]
        sd[i].values_p[0,:, 0,:,0] = sd[i].halo_p[4,1:-1,1:-1]
        sd[i].values_p[0,:,-1,:,0] = sd[i].halo_p[5,1:-1,1:-1]
        sd[i].values_p[0,:,:, 0,0] = sd[i].halo_p[2,1:-1,1:-1]
        sd[i].values_p[0,:,:,-1,0] = sd[i].halo_p[3,1:-1,1:-1]



# def bluff_body(sd,sigma,dt,xmin,xmax,ymin,ymax,zmin,zmax):
#     sd.values_u[0,zmin:zmax,ymin:ymax,xmin:xmax,0] = sd.values_u[0,zmin:zmax,ymin:ymax,xmin:xmax,0]/(1+dt*sigma)
#     sd.values_u[0,zmin:zmax,ymin:ymax,xmin:xmax,0] = sd.values_u[0,zmin:zmax,ymin:ymax,xmin:xmax,0]/(1+dt*sigma)
#     sd.values_u[0,zmin:zmax,ymin:ymax,xmin:xmax,0] = sd.values_u[0,zmin:zmax,ymin:ymax,xmin:xmax,0]/(1+dt*sigma)


def bluff_body(sd,dt):
    for i in range(len(sd)):
        sd[i].values_u /= (1+dt*sd[i].sigma)
        sd[i].values_u /= (1+dt*sd[i].sigma)
        sd[i].values_u /= (1+dt*sd[i].sigma)



def generate_points_and_cells(num_nodes_x, num_nodes_y):
    '''
    it saves 2D points and cells making them ready to be fed into vtk creator. 
    '''
    # Generate coordinates
    points = []
    for j in range(num_nodes_y):
        for i in range(num_nodes_x):
            points.append([i, j, 0])

    # Generate cell connectivity
    cells = []
    for j in range(num_nodes_y - 1):
        for i in range(num_nodes_x - 1):
            cell = [
                j * num_nodes_x + i,
                j * num_nodes_x + i + 1,
                (j + 1) * num_nodes_x + i + 1,
                (j + 1) * num_nodes_x + i
            ]
            cells.append(cell)

    return points, cells




def generate_vtu_file(points, cells, values, output_file):
    # Create a VTK unstructured grid
    unstructured_grid = vtk.vtkUnstructuredGrid()

    # Set points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)
    unstructured_grid.SetPoints(vtk_points)

    # Set cell connectivity
    vtk_cells = vtk.vtkCellArray()
    for cell in cells:
        vtk_cells.InsertNextCell(len(cell), cell)
    unstructured_grid.SetCells(vtk.VTK_QUAD, vtk_cells)

    # Set point values
    vtk_data = vtk.vtkFloatArray()
    for value in values:
        vtk_data.InsertNextValue(value)
    unstructured_grid.GetPointData().SetScalars(vtk_data)

    # Write the VTU file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(unstructured_grid)
    writer.Write()

    

def scale_faces(my_values, values):
    '''
    values2 = scale(values)
    the input value is a 2D representing shared faces
    interpolates values of adjacent faces in 3D from high resolution to low resolution by averaging 4 nodes into 1
    the aspect ration must be 2:1
    '''
    # high res to low res
    ratio = np.shape(my_values[0,:])[0] / np.shape(values[0,:])[0]
    
    if (ratio) < 1:
        # it gets average of 4 points and put it into one coarse point
        rows, cols = values.shape
        rows, cols = rows // 2, cols // 2
        array2 = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                array2[i, j] = 1/4 * (values[i*2, j*2] + values[i*2, j*2+1] + values[i*2+1, j*2] + values[i*2+1, j*2+1])
    
        return array2

    elif ratio == 1:
        return values
    

    elif ratio > 1:
        # it set 4 points in the high resolution area equal to one point in the low resolution
        array2 = np.zeros((2 * values.shape[0], 2 * values.shape[0]))
        for i in range(values.shape[0]):
            for j in range(values.shape[0]):
                array2[2*i, 2*j] = values[i, j]
                array2[2*i, 2*j+1] = values[i, j]
                array2[2*i+1, 2*j] = values[i, j]
                array2[2*i+1, 2*j+1] = values[i, j]
        
        return array2
    


def scale_sigma(sigma, sd):
    ratio = np.shape(sd.sigma[0,:,0,0,0])[0] / np.shape(sigma[0,:,0,0,0])[0]
    
    if ratio <1:
        nz, ny, nx = sigma.shape
        nz, ny, nx = nz // 2, ny // 2, nx // 2
        array3D = np.zeros((nz, ny, nx))
        
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    array3D[i, j, k] = (
                        sigma[i*2, j*2, k*2] + sigma[i*2, j*2+1, k*2] + sigma[i*2+1, j*2, k*2] + sigma[i*2+1, j*2+1, k*2] +
                        sigma[i*2, j*2, k*2+1] + sigma[i*2, j*2+1, k*2+1] + sigma[i*2+1, j*2, k*2+1] + sigma[i*2+1, j*2+1, k*2+1]
                    ) / 8
        
        return array3D
    
    elif ratio == 1:
        return sigma
    
        



    
    


def scale_arrays(my_values, neig_values):
    '''
    values2 = scale(values)
    the input value is a 1D representing corner arrays
    interpolates values of adjacent array in 3D from high resolution to low resolution by averaging 2 nodes into 1
    the aspect ration must be 2:1
    '''
    # high res to low res
    ratio = np.shape(my_values)[0] / np.shape(neig_values)[0]
    
    if (ratio) < 1:
        '''ONLY for even numbers of nx, ny. it interpolates nodes from low to high with a ratio of 2:1'''
        # Reshape 'a' into a 2D array with two columns
        arr_2d = neig_values.reshape(-1, 2)

        # Compute the average of each pair of values along the rows
        scaled_down = np.mean(arr_2d, axis=1)

        return scaled_down

    elif ratio == 1:
        return neig_values
    


    elif ratio > 1:
        scaled_up = np.repeat(neig_values,2)
        
        return scaled_up
    


def add_padding_u(sd, values_u):
    padded_array = np.zeros((1,sd.nz + 2, sd.ny + 2 , sd.nx + 2,1))
    padded_array[:, 1:-1, 1:-1, 1:-1, :] = values_u
    
    padded_array[0,0,:,:,0]   = sd.halo_u[0,:,:]                # front
    padded_array[0,-1,:,:,0]  = sd.halo_u[1,:,:]                # back
    padded_array[0,:,:,0,0]   = sd.halo_u[2,:,:]                # left
    padded_array[0,:,:,-1,0]  = sd.halo_u[3,:,:]                # right
    padded_array[0,:,0,:,0]   = sd.halo_u[4,:,:]                # top
    padded_array[0,:,-1,:,0]  = sd.halo_u[5,:,:]                # bottom
    # corner arrays 
    # here it is a bit generalised as I have assumed that iface == 2 is always 'inlet' and iface=3 is 'outlet' 
    for iface in [0,1,4,5,2,3]: # check the in/outlet last to make sure they are set to 1
        if sd.neig[iface] == 'outlet': # iface == 3
            padded_array[0, :,    0, -1, 0] = 1 #sd.values_u[0,  :,  1, -2, 0]       # top
            padded_array[0, :,   -1, -1, 0] = 1 #sd.values_u[0,  :, -2, -2, 0]       # bottom
            padded_array[0,   -1, :, -1, 0] = 1 #sd.values_u[0, -2,  :, -2, 0]       # left
            padded_array[0,    0, :, -1, 0] = 1 #sd.values_u[0,  1,  :, -2, 0]       # right

        
        elif sd.neig[iface] == 'inlet': # iface == 2
            padded_array[0, :,    0, 0, 0] = 1 #sd.values_u[0,  :,  1, 1, 0]         # top
            padded_array[0, :,   -1, 0, 0] = 1 #sd.values_u[0,  :, -2, 1, 0]         # bottom
            padded_array[0,   -1, :, 0, 0] = 1 #sd.values_u[0, -2,  :, 1, 0]         # left
            padded_array[0,    0, :, 0, 0] = 1 #sd.values_u[0,  1,  :, 1, 0]         # right    

        else:
            if sd.neig[iface] == 'side':
                if iface == 0:
                    padded_array[0, 0,    0, 1:-1, 0] = sd.values_u[0, 1, 1, :, 0]           # top
                    padded_array[0, 0,   -1, 1:-1, 0] = sd.values_u[0, 1,-2, :, 0]           # bottom
                    padded_array[0, 0, 1:-1,    0, 0] = sd.values_u[0, 1, :, 1, 0]           # left
                    padded_array[0, 0, 1:-1,   -1, 0] = sd.values_u[0, 1, :,-2, 0]           # right
                elif iface == 1:
                    padded_array[0, -1,    0, 1:-1, 0] = sd.values_u[0,-2, 1, :, 0]          # top
                    padded_array[0, -1,   -1, 1:-1, 0] = sd.values_u[0,-2,-2, :, 0]          # bottom
                    padded_array[0, -1, 1:-1,    0, 0] = sd.values_u[0,-2, :, 1, 0]          # left
                    padded_array[0, -1, 1:-1,   -1, 0] = sd.values_u[0,-2, :,-2, 0]          # right
                elif iface == 2:
                    padded_array[0, 1:-1,    0, 0, 0] = sd.values_u[0,  :,  1, 1, 0]         # top
                    padded_array[0, 1:-1,   -1, 0, 0] = sd.values_u[0,  :, -2, 1, 0]         # bottom
                    padded_array[0,   -1, 1:-1, 0, 0] = sd.values_u[0, -2,  :, 1, 0]         # left
                    padded_array[0,    0, 1:-1, 0, 0] = sd.values_u[0,  1,  :, 1, 0]         # right
                elif iface == 3:
                    padded_array[0, 1:-1,    0, -1, 0] = sd.values_u[0,  :,  1, -2, 0]       # top
                    padded_array[0, 1:-1,   -1, -1, 0] = sd.values_u[0,  :, -2, -2, 0]       # bottom
                    padded_array[0,   -1, 1:-1, -1, 0] = sd.values_u[0, -2,  :, -2, 0]       # left
                    padded_array[0,    0, 1:-1, -1, 0] = sd.values_u[0,  1,  :, -2, 0]       # right
                elif iface == 4:
                    padded_array[0,    -1, 0, 1:-1, 0] = sd.values_u[0, -2, 1,  :, 0]        # top
                    padded_array[0,     0, 0, 1:-1, 0] = sd.values_u[0,  1, 1,  :, 0]        # bottom
                    padded_array[0,  1:-1, 0,    0, 0] = sd.values_u[0,  :, 1,  1, 0]        # left
                    padded_array[0,  1:-1, 0,   -1, 0] = sd.values_u[0,  :, 1, -2, 0]        # right
                elif iface == 5:
                    padded_array[0,   -1, -1, 1:-1, 0] = sd.values_u[0, -2, -2,  :, 0]       # top
                    padded_array[0,    0, -1, 1:-1, 0] = sd.values_u[0,  1, -2,  :, 0]       # bottom
                    padded_array[0, 1:-1, -1,    0, 0] = sd.values_u[0,  :, -2,  1, 0]       # left
                    padded_array[0, 1:-1, -1,   -1, 0] = sd.values_u[0,  :, -2, -2, 0]       # right

            else:#elif sd.neig[iface] != 'inlet' and sd.neig[iface] != 'outlet': # means it has a neighbouring subdomain
                if iface == 0:
                    padded_array[0, 0,    0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_u[0, -1, 0, :, 0])         # top
                    padded_array[0, 0,   -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_u[0, -1,-1, :, 0])        # bottom
                    padded_array[0, 0, 1:-1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_u[0, -1, :, 0, 0])         # left
                    padded_array[0, 0, 1:-1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_u[0, -1, :,-1, 0])         # right
                elif iface == 1:
                    padded_array[0, -1,    0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_u[0,0, 0, :, 0])          # top
                    padded_array[0, -1,   -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_u[0,0,-1, :, 0])          # bottom
                    padded_array[0, -1, 1:-1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_u[0,0, :, 0, 0])          # left
                    padded_array[0, -1, 1:-1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_u[0,0, :,-1, 0])          # right
                elif iface == 2:
                    padded_array[0, 1:-1,    0, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_u[0,  :,  0, -1, 0])       # top
                    padded_array[0, 1:-1,   -1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_u[0,  :, -1, -1, 0])       # bottom
                    padded_array[0,   -1, 1:-1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_u[0, -1,  :, -1, 0])       # left
                    padded_array[0,    0, 1:-1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_u[0,  0,  :, -1, 0])       # right
                elif iface == 3:
                    padded_array[0, 1:-1,    0, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_u[0,  :, 0,0,0])          # top
                    padded_array[0, 1:-1,   -1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_u[0,  :,-1,0,0])          # bottom
                    padded_array[0,   -1, 1:-1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_u[0, -1, :,0,0])          # left
                    padded_array[0,    0, 1:-1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_u[0,  0, :,0,0])          # right
                elif iface == 4:
                    padded_array[0,    -1, 0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_u[0, -1, -1,  :, 0])      # top
                    padded_array[0,     0, 0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_u[0,  0, -1,  :, 0])      # bottom
                    padded_array[0,  1:-1, 0,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_u[0,  :, -1,  0, 0])     # left
                    padded_array[0,  1:-1, 0,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_u[0,  :, -1, -1, 0])      # right
                elif iface == 5:
                    padded_array[0,   -1, -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_u[0, -1, 0,  :, 0])       # top
                    padded_array[0,    0, -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_u[0,  0, 0,  :, 0])       # bottom
                    padded_array[0, 1:-1, -1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_u[0,  :, 0,  0, 0])       # left
                    padded_array[0, 1:-1, -1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_u[0,  :, 0, -1, 0])       # right
                    
    # corner points
    if sd.corner_neig[0] == 'No':                                                       # node 0 == 6 of corner_neig
        padded_array[0,  0, -1,  0, 0] = sd.values_u[0,  1, -2,  1, 0] 
    elif not isinstance(sd.corner_neig[0], str):   
        padded_array[0,  0, -1,  0, 0] = sd.corner_neig[0].values_u[0, -2,1,-2, 0]
    
    if sd.corner_neig[1] == 'No':                                                        # node 1 == 7 of corner_neig
        padded_array[0,  0, -1, -1, 0] = sd.values_u[0,  1, -2, -2, 0]
    elif not isinstance(sd.corner_neig[1], str):
        padded_array[0,  0, -1, -1, 0] = sd.corner_neig[1].values_u[0, -2,1,1, 0]

    if sd.corner_neig[2] == 'No':                                                        # node 2 == 4 of corner_neig
        padded_array[0,  0,  0, -1, 0] = sd.values_u[0,  1,  1, -2, 0]
    elif not isinstance(sd.corner_neig[2], str):
        padded_array[0,  0,  0, -1, 0] = sd.corner_neig[2].values_u[0, -2,-2,1, 0]

    if sd.corner_neig[3] == 'No':                                                        # node 3 == 5 of corner_neig
        padded_array[0,  0,  0,  0, 0] = sd.values_u[0,  1,  1,  1, 0]
    elif not isinstance(sd.corner_neig[3], str):
        padded_array[0,  0,  0,  0, 0] = sd.corner_neig[3].values_u[0, -2,-2,-2, 0]

    if sd.corner_neig[4] == 'No':                                                        # node 4 == 2 of corner_neig
        padded_array[0, -1, -1,  0, 0] = sd.values_u[0, -2, -2,  1, 0]
    elif not isinstance(sd.corner_neig[4], str):
        padded_array[0, -1, -1,  0, 0] = sd.corner_neig[4].values_u[0, 1,1,-2, 0]

    if sd.corner_neig[5] == 'No':                                                        # node 5 == 3 of corner_neig
        padded_array[0, -1, -1, -1, 0] = sd.values_u[0, -2, -2, -2, 0]
    elif not isinstance(sd.corner_neig[5], str):
        padded_array[0, -1, -1, -1, 0] = sd.corner_neig[5].values_u[0, 1,1,1, 0]

    if sd.corner_neig[6] == 'No':                                                        # node 6 == 0 of corner_neig
        padded_array[0, -1,  0, -1, 0] = sd.values_u[0, -2,  1, -2, 0]
    elif not isinstance(sd.corner_neig[6], str):
        padded_array[0, -1,  0, -1, 0] = sd.corner_neig[6].values_u[0, 1,-2, 1, 0]

    if sd.corner_neig[7] == 'No':                                                        # node 7 == 1 of corner_neig
        padded_array[0, -1,  0,  0, 0] = sd.values_u[0, -2,  1,  1, 0]
    elif not isinstance(sd.corner_neig[7], str):
        padded_array[0, -1,  0,  0, 0] = sd.corner_neig[7].values_u[0, 1,-2,-2, 0]

    return padded_array


def add_padding_v(sd, values_v):
    padded_array = np.zeros((1,sd.nz + 2, sd.ny + 2 , sd.nx + 2,1))
    padded_array[:, 1:-1, 1:-1, 1:-1, :] = values_v
    
    padded_array[0,0,:,:,0]   = sd.halo_v[0,:,:]                # front
    padded_array[0,-1,:,:,0]  = sd.halo_v[1,:,:]                # back
    padded_array[0,:,:,0,0]   = sd.halo_v[2,:,:]                # left
    padded_array[0,:,:,-1,0]  = sd.halo_v[3,:,:]                # right
    padded_array[0,:,0,:,0]   = sd.halo_v[4,:,:]                # top
    padded_array[0,:,-1,:,0]  = sd.halo_v[5,:,:]                # bottom
    
    # corner arrays 
    for iface in [0,1,4,5,2,3]: # check the in/outlet/sides last to make sure they are set to 0
        if sd.neig[iface] == 'inlet': # iface == 2
            padded_array[0, :,    0, -1, 0] = 0 #sd.values_v[0,  :,  1, -2, 0]       # top
            padded_array[0, :,   -1, -1, 0] = 0 #sd.values_v[0,  :, -2, -2, 0]       # bottom
            padded_array[0,   -1, :, -1, 0] = 0 #sd.values_v[0, -2,  :, -2, 0]       # left
            padded_array[0,    0, :, -1, 0] = 0 #sd.values_v[0,  1,  :, -2, 0]       # right
        elif sd.neig[iface] == 'outlet': # iface == 3
            padded_array[0, :,    0, 0, 0] = 0 #sd.values_v[0,  :,  1, 1, 0]         # top
            padded_array[0, :,   -1, 0, 0] = 0 #sd.values_v[0,  :, -2, 1, 0]         # bottom
            padded_array[0,   -1, :, 0, 0] = 0 #sd.values_v[0, -2,  :, 1, 0]         # left
            padded_array[0,    0, :, 0, 0] = 0 #sd.values_v[0,  1,  :, 1, 0]         # right
        else:
            if sd.neig[iface] != 'side':# and sd.neig[iface] != 'inlet' and sd.neig[iface] != 'outlet': # means it has a neighbouring subdomain
                if iface == 0:
                    padded_array[0, 0,    0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_v[0, -1, 0, :, 0])         # top
                    padded_array[0, 0,   -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_v[0, -1,-1, :, 0])         # bottom
                    padded_array[0, 0, 1:-1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_v[0, -1, :, 0, 0])         # left
                    padded_array[0, 0, 1:-1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_v[0, -1, :,-1, 0])         # right
                elif iface == 1:
                    padded_array[0, -1,    0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_v[0,0, 0, :, 0])          # top
                    padded_array[0, -1,   -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_v[0,0,-1, :, 0])          # bottom
                    padded_array[0, -1, 1:-1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_v[0,0, :, 0, 0])          # left
                    padded_array[0, -1, 1:-1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_v[0,0, :,-1, 0])          # right
                elif iface == 2:
                    padded_array[0, 1:-1,    0, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_v[0,  :,  0, -1, 0])       # top
                    padded_array[0, 1:-1,   -1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_v[0,  :, -1, -1, 0])       # bottom
                    padded_array[0,   -1, 1:-1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_v[0, -1,  :, -1, 0])       # left
                    padded_array[0,    0, 1:-1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_v[0,  0,  :, -1, 0])       # right
                elif iface == 3:
                    padded_array[0, 1:-1,    0, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_v[0,  :, 0,0,0])          # top
                    padded_array[0, 1:-1,   -1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_v[0,  :,-1,0,0])          # bottom
                    padded_array[0,   -1, 1:-1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_v[0, -1, :,0,0])          # left
                    padded_array[0,    0, 1:-1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_v[0,  0, :,0,0])          # right
                elif iface == 4:
                    padded_array[0,    -1, 0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_v[0, -1, -1,  :, 0])      # top
                    padded_array[0,     0, 0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_v[0,  0, -1,  :, 0])      # bottom
                    padded_array[0,  1:-1, 0,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_v[0,  :, -1,  0, 0])      # left
                    padded_array[0,  1:-1, 0,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_v[0,  :, -1, -1, 0])      # right
                elif iface == 5:
                    padded_array[0,   -1, -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_v[0, -1, 0,  :, 0])       # top
                    padded_array[0,    0, -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_v[0,  0, 0,  :, 0])       # bottom
                    padded_array[0, 1:-1, -1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_v[0,  :, 0,  0, 0])       # left
                    padded_array[0, 1:-1, -1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_v[0,  :, 0, -1, 0])       # right

    # kept it separate to make sure all physical bc are zero
    for i in range(6):           
        if sd.neig[i] == 'side': # means on the physical boundaries
            if iface == 0:
                padded_array[0, 0,    0, :, 0] = 0 #sd.values_v[0, 1, 1, :, 0]           # top
                padded_array[0, 0,   -1, :, 0] = 0 #sd.values_v[0, 1,-2, :, 0]           # bottom
                padded_array[0, 0, :,    0, 0] = 0 #sd.values_v[0, 1, :, 1, 0]           # left
                padded_array[0, 0, :,   -1, 0] = 0 #sd.values_v[0, 1, :,-2, 0]           # right
            elif iface == 1:
                padded_array[0, -1,    0, :, 0] = 0 #sd.values_v[0,-2, 1, :, 0]          # top
                padded_array[0, -1,   -1, :, 0] = 0 #sd.values_v[0,-2,-2, :, 0]          # bottom
                padded_array[0, -1, :,    0, 0] = 0 #sd.values_v[0,-2, :, 1, 0]          # left
                padded_array[0, -1, :,   -1, 0] = 0 #sd.values_v[0,-2, :,-2, 0]          # right
            elif iface == 2:
                padded_array[0, :,    0, 0, 0] = 0 #sd.values_v[0,  :,  1, 1, 0]         # top
                padded_array[0, :,   -1, 0, 0] = 0 #sd.values_v[0,  :, -2, 1, 0]         # bottom
                padded_array[0,   -1, :, 0, 0] = 0 #sd.values_v[0, -2,  :, 1, 0]         # left
                padded_array[0,    0, :, 0, 0] = 0 #sd.values_v[0,  1,  :, 1, 0]         # right
            elif iface == 3:
                padded_array[0, :,    0, -1, 0] = 0 #sd.values_v[0,  :,  1, -2, 0]       # top
                padded_array[0, :,   -1, -1, 0] = 0 #sd.values_v[0,  :, -2, -2, 0]       # bottom
                padded_array[0,   -1, :, -1, 0] = 0 #sd.values_v[0, -2,  :, -2, 0]       # left
                padded_array[0,    0, :, -1, 0] = 0 #sd.values_v[0,  1,  :, -2, 0]       # right
            elif iface == 4:
                padded_array[0,    -1, 0, :, 0] = 0 #sd.values_v[0, -2, 1,  :, 0]        # top
                padded_array[0,     0, 0, :, 0] = 0 #sd.values_v[0,  1, 1,  :, 0]        # bottom
                padded_array[0,  :, 0,    0, 0] = 0 #sd.values_v[0,  :, 1,  1, 0]        # left
                padded_array[0,  :, 0,   -1, 0] = 0 #sd.values_v[0,  :, 1, -2, 0]        # right
            elif iface == 5:
                padded_array[0,   -1, -1, :, 0] = 0 #sd.values_v[0, -2, -2,  :, 0]       # top
                padded_array[0,    0, -1, :, 0] = 0 #sd.values_v[0,  1, -2,  :, 0]       # bottom
                padded_array[0, :, -1,    0, 0] = 0 #sd.values_v[0,  :, -2,  1, 0]       # left
                padded_array[0, :, -1,   -1, 0] = 0 #sd.values_v[0,  :, -2, -2, 0]       # right

    # corner points
    if sd.corner_neig[0] == 'No':                                                       # node 0 == 6 of corner_neig
        padded_array[0,  0, -1,  0, 0] = 0 #sd.values_v[0,  1, -2,  1, 0] 
    elif not isinstance(sd.corner_neig[0], str):   
        padded_array[0,  0, -1,  0, 0] = sd.corner_neig[0].values_v[0, -2,1,-2, 0]
    
    if sd.corner_neig[1] == 'No':                                                        # node 1 == 7 of corner_neig
        padded_array[0,  0, -1, -1, 0] = 0 #sd.values_v[0,  1, -2, -2, 0]
    elif not isinstance(sd.corner_neig[1], str):
        padded_array[0,  0, -1, -1, 0] = sd.corner_neig[1].values_v[0, -2,1,1, 0]

    if sd.corner_neig[2] == 'No':                                                        # node 2 == 4 of corner_neig
        padded_array[0,  0,  0, -1, 0] = 0 #sd.values_v[0,  1,  1, -2, 0]
    elif not isinstance(sd.corner_neig[2], str):
        padded_array[0,  0,  0, -1, 0] = sd.corner_neig[2].values_v[0, -2,-2,1, 0]

    if sd.corner_neig[3] == 'No':                                                        # node 3 == 5 of corner_neig
        padded_array[0,  0,  0,  0, 0] = 0 #sd.values_v[0,  1,  1,  1, 0]
    elif not isinstance(sd.corner_neig[3], str):
        padded_array[0,  0,  0,  0, 0] = sd.corner_neig[3].values_v[0, -2,-2,-2, 0]

    if sd.corner_neig[4] == 'No':                                                        # node 4 == 2 of corner_neig
        padded_array[0, -1, -1,  0, 0] = 0 #sd.values_v[0, -2, -2,  1, 0]
    elif not isinstance(sd.corner_neig[4], str):
        padded_array[0, -1, -1,  0, 0] = sd.corner_neig[4].values_v[0, 1,1,-2, 0]

    if sd.corner_neig[5] == 'No':                                                        # node 5 == 3 of corner_neig
        padded_array[0, -1, -1, -1, 0] = 0 #sd.values_v[0, -2, -2, -2, 0]
    elif not isinstance(sd.corner_neig[5], str):
        padded_array[0, -1, -1, -1, 0] = sd.corner_neig[5].values_v[0, 1,1,1, 0]

    if sd.corner_neig[6] == 'No':                                                        # node 6 == 0 of corner_neig
        padded_array[0, -1,  0, -1, 0] = 0 #sd.values_v[0, -2,  1, -2, 0]
    elif not isinstance(sd.corner_neig[6], str):
        padded_array[0, -1,  0, -1, 0] = sd.corner_neig[6].values_v[0, 1,-2, 1, 0]

    if sd.corner_neig[7] == 'No':                                                        # node 7 == 1 of corner_neig
        padded_array[0, -1,  0,  0, 0] = 0 #sd.values_v[0, -2,  1,  1, 0]
    elif not isinstance(sd.corner_neig[7], str):
        padded_array[0, -1,  0,  0, 0] = sd.corner_neig[7].values_v[0, 1,-2,-2, 0]

    return padded_array


def add_padding_w(sd, values_w):
    padded_array = np.zeros((1,sd.nz + 2, sd.ny + 2 , sd.nx + 2,1))
    padded_array[:, 1:-1, 1:-1, 1:-1, :] = values_w
    
    padded_array[0,0,:,:,0]   = sd.halo_w[0,:,:]                # front
    padded_array[0,-1,:,:,0]  = sd.halo_w[1,:,:]                # back
    padded_array[0,:,:,0,0]   = sd.halo_w[2,:,:]                # left
    padded_array[0,:,:,-1,0]  = sd.halo_w[3,:,:]                # right
    padded_array[0,:,0,:,0]   = sd.halo_w[4,:,:]                # top
    padded_array[0,:,-1,:,0]  = sd.halo_w[5,:,:]                # bottom
    # corner arrays 
    for iface in [0,1,4,5,2,3]: # check the in/outlet/sides last to make sure they are set to 0
        if sd.neig[iface] == 'inlet': # iface == 2
            padded_array[0, :,    0, -1, 0] = 0 #sd.values_w[0,  :,  1, -2, 0]       # top
            padded_array[0, :,   -1, -1, 0] = 0 #sd.values_w[0,  :, -2, -2, 0]       # bottom
            padded_array[0,   -1, :, -1, 0] = 0 #sd.values_w[0, -2,  :, -2, 0]       # left
            padded_array[0,    0, :, -1, 0] = 0 #sd.values_w[0,  1,  :, -2, 0]       # right
        elif sd.neig[iface] == 'outlet': # iface == 3
            padded_array[0, :,    0, 0, 0] = 0 #sd.values_w[0,  :,  1, 1, 0]         # top
            padded_array[0, :,   -1, 0, 0] = 0 #sd.values_w[0,  :, -2, 1, 0]         # bottom
            padded_array[0,   -1, :, 0, 0] = 0 #sd.values_w[0, -2,  :, 1, 0]         # left
            padded_array[0,    0, :, 0, 0] = 0 #sd.values_w[0,  1,  :, 1, 0]         # right
        else:
            if sd.neig[iface] != 'side' and sd.neig[iface] != 'inlet' and sd.neig[iface] != 'outlet': # means it has a neighbouring subdomain
                if iface == 0:
                    padded_array[0, 0,    0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_w[0, -1, 0, :, 0])         # top
                    padded_array[0, 0,   -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_w[0, -1,-1, :, 0])         # bottom
                    padded_array[0, 0, 1:-1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_w[0, -1, :, 0, 0])         # left
                    padded_array[0, 0, 1:-1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_w[0, -1, :,-1, 0])         # right
                elif iface == 1:
                    padded_array[0, -1,    0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_w[0,0, 0, :, 0])          # top
                    padded_array[0, -1,   -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_w[0,0,-1, :, 0])          # bottom
                    padded_array[0, -1, 1:-1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_w[0,0, :, 0, 0])          # left
                    padded_array[0, -1, 1:-1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_w[0,0, :,-1, 0])          # right
                elif iface == 2:
                    padded_array[0, 1:-1,    0, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_w[0,  :,  0, -1, 0])       # top
                    padded_array[0, 1:-1,   -1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_w[0,  :, -1, -1, 0])       # bottom
                    padded_array[0,   -1, 1:-1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_w[0, -1,  :, -1, 0])       # left
                    padded_array[0,    0, 1:-1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_w[0,  0,  :, -1, 0])       # right
                elif iface == 3:
                    padded_array[0, 1:-1,    0, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_w[0,  :, 0,0,0])          # top
                    padded_array[0, 1:-1,   -1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_w[0,  :,-1,0,0])          # bottom
                    padded_array[0,   -1, 1:-1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_w[0, -1, :,0,0])          # left
                    padded_array[0,    0, 1:-1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_w[0,  0, :,0,0])          # right
                elif iface == 4:
                    padded_array[0,    -1, 0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_w[0, -1, -1,  :, 0])      # top
                    padded_array[0,     0, 0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_w[0,  0, -1,  :, 0])      # bottom
                    padded_array[0,  1:-1, 0,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_w[0,  :, -1,  0, 0])      # left
                    padded_array[0,  1:-1, 0,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_w[0,  :, -1, -1, 0])      # right
                elif iface == 5:
                    padded_array[0,   -1, -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_w[0, -1, 0,  :, 0])       # top
                    padded_array[0,    0, -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_w[0,  0, 0,  :, 0])       # bottom
                    padded_array[0, 1:-1, -1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_w[0,  :, 0,  0, 0])       # left
                    padded_array[0, 1:-1, -1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_w[0,  :, 0, -1, 0])       # right
    
    # kept it separate to make sure all physical bc are zero
    for i in range(6):           
        if sd.neig[i] == 'side': # means on the physical boundaries
            if iface == 0:
                padded_array[0, 0,    0, :, 0] = 0 #sd.values_w[0, 1, 1, :, 0]           # top
                padded_array[0, 0,   -1, :, 0] = 0 #sd.values_w[0, 1,-2, :, 0]           # bottom
                padded_array[0, 0, :,    0, 0] = 0 #sd.values_w[0, 1, :, 1, 0]           # left
                padded_array[0, 0, :,   -1, 0] = 0 #sd.values_w[0, 1, :,-2, 0]           # right
            elif iface == 1:
                padded_array[0, -1,    0, :, 0] = 0 #sd.values_w[0,-2, 1, :, 0]          # top
                padded_array[0, -1,   -1, :, 0] = 0 #sd.values_w[0,-2,-2, :, 0]          # bottom
                padded_array[0, -1, :,    0, 0] = 0 #sd.values_w[0,-2, :, 1, 0]          # left
                padded_array[0, -1, :,   -1, 0] = 0 #sd.values_w[0,-2, :,-2, 0]          # right
            elif iface == 2:
                padded_array[0, :,    0, 0, 0] = 0 #sd.values_w[0,  :,  1, 1, 0]         # top
                padded_array[0, :,   -1, 0, 0] = 0 #sd.values_w[0,  :, -2, 1, 0]         # bottom
                padded_array[0,   -1, :, 0, 0] = 0 #sd.values_w[0, -2,  :, 1, 0]         # left
                padded_array[0,    0, :, 0, 0] = 0 #sd.values_w[0,  1,  :, 1, 0]         # right
            elif iface == 3:
                padded_array[0, :,    0, -1, 0] = 0 #sd.values_w[0,  :,  1, -2, 0]       # top
                padded_array[0, :,   -1, -1, 0] = 0 #sd.values_w[0,  :, -2, -2, 0]       # bottom
                padded_array[0,   -1, :, -1, 0] = 0 #sd.values_w[0, -2,  :, -2, 0]       # left
                padded_array[0,    0, :, -1, 0] = 0 #sd.values_w[0,  1,  :, -2, 0]       # right
            elif iface == 4:
                padded_array[0,    -1, 0, :, 0] = 0 #sd.values_w[0, -2, 1,  :, 0]        # top
                padded_array[0,     0, 0, :, 0] = 0 #sd.values_w[0,  1, 1,  :, 0]        # bottom
                padded_array[0,  :, 0,    0, 0] = 0 #sd.values_w[0,  :, 1,  1, 0]        # left
                padded_array[0,  :, 0,   -1, 0] = 0 #sd.values_w[0,  :, 1, -2, 0]        # right
            elif iface == 5:
                padded_array[0,   -1, -1, :, 0] = 0 #sd.values_w[0, -2, -2,  :, 0]       # top
                padded_array[0,    0, -1, :, 0] = 0 #sd.values_w[0,  1, -2,  :, 0]       # bottom
                padded_array[0, :, -1,    0, 0] = 0 #sd.values_w[0,  :, -2,  1, 0]       # left
                padded_array[0, :, -1,   -1, 0] = 0 #sd.values_=w[0,  :, -2, -2, 0]       # right

    # corner points
    if sd.corner_neig[0] == 'No':                                                       # node 0 == 6 of corner_neig
        padded_array[0,  0, -1,  0, 0] = 0 #sd.values_w[0,  1, -2,  1, 0] 
    elif not isinstance(sd.corner_neig[0], str):   
        padded_array[0,  0, -1,  0, 0] = sd.corner_neig[0].values_w[0, -2,1,-2, 0]
    
    if sd.corner_neig[1] == 'No':                                                        # node 1 == 7 of corner_neig
        padded_array[0,  0, -1, -1, 0] = 0 #sd.values_w[0,  1, -2, -2, 0]
    elif not isinstance(sd.corner_neig[1], str):
        padded_array[0,  0, -1, -1, 0] = sd.corner_neig[1].values_w[0, -2,1,1, 0]

    if sd.corner_neig[2] == 'No':                                                        # node 2 == 4 of corner_neig
        padded_array[0,  0,  0, -1, 0] = 0 #sd.values_w[0,  1,  1, -2, 0]
    elif not isinstance(sd.corner_neig[2], str):
        padded_array[0,  0,  0, -1, 0] = sd.corner_neig[2].values_w[0, -2,-2,1, 0]

    if sd.corner_neig[3] == 'No':                                                        # node 3 == 5 of corner_neig
        padded_array[0,  0,  0,  0, 0] = 0 #sd.values_w[0,  1,  1,  1, 0]
    elif not isinstance(sd.corner_neig[3], str):
        padded_array[0,  0,  0,  0, 0] = sd.corner_neig[3].values_w[0, -2,-2,-2, 0]

    if sd.corner_neig[4] == 'No':                                                        # node 4 == 2 of corner_neig
        padded_array[0, -1, -1,  0, 0] = 0 #sd.values_w[0, -2, -2,  1, 0]
    elif not isinstance(sd.corner_neig[4], str):
        padded_array[0, -1, -1,  0, 0] = sd.corner_neig[4].values_w[0, 1,1,-2, 0]

    if sd.corner_neig[5] == 'No':                                                        # node 5 == 3 of corner_neig
        padded_array[0, -1, -1, -1, 0] = 0 #sd.values_w[0, -2, -2, -2, 0]
    elif not isinstance(sd.corner_neig[5], str):
        padded_array[0, -1, -1, -1, 0] = sd.corner_neig[5].values_w[0, 1,1,1, 0]

    if sd.corner_neig[6] == 'No':                                                        # node 6 == 0 of corner_neig
        padded_array[0, -1,  0, -1, 0] = 0 #sd.values_w[0, -2,  1, -2, 0]
    elif not isinstance(sd.corner_neig[6], str):
        padded_array[0, -1,  0, -1, 0] = sd.corner_neig[6].values_w[0, 1,-2, 1, 0]

    if sd.corner_neig[7] == 'No':                                                        # node 7 == 1 of corner_neig
        padded_array[0, -1,  0,  0, 0] = 0 #sd.values_w[0, -2,  1,  1, 0]
    elif not isinstance(sd.corner_neig[7], str):
        padded_array[0, -1,  0,  0, 0] = sd.corner_neig[7].values_w[0, 1,-2,-2, 0]

    return padded_array


def add_padding_p(sd):
    padded_array = np.zeros((1,sd.nz + 2, sd.ny + 2 , sd.nx + 2,1))
    padded_array[:, 1:-1, 1:-1, 1:-1, :] = sd.values_p
    
    padded_array[0,0,:,:,0]   = sd.halo_p[0,:,:]                # front
    padded_array[0,-1,:,:,0]  = sd.halo_p[1,:,:]                # back
    padded_array[0,:,:,0,0]   = sd.halo_p[2,:,:]                # left
    padded_array[0,:,:,-1,0]  = sd.halo_p[3,:,:]                # right
    padded_array[0,:,0,:,0]   = sd.halo_p[4,:,:]                # top
    padded_array[0,:,-1,:,0]  = sd.halo_p[5,:,:]                # bottom
    # corner arrays
    # here it is a bit generalised as I have assumed that iface == 2 is always 'inlet' and iface=3 is 'outlet' 
    for iface in [0,1,2,4,5,3]: # check the outlet last to make sure it sets all values to 0
        if sd.neig[iface] == 'outlet': 
            padded_array[0, :, :, -1, 0] = 0
            
        else:
            if sd.neig[iface] == 'side' or sd.neig[iface] == 'inlet':
                if iface == 0:
                    padded_array[0, 0,    0, 1:-1, 0] = sd.values_p[0, 1, 1, :, 0]           # top
                    padded_array[0, 0,   -1, 1:-1, 0] = sd.values_p[0, 1,-2, :, 0]           # bottom
                    padded_array[0, 0, 1:-1,    0, 0] = sd.values_p[0, 1, :, 1, 0]           # left
                    padded_array[0, 0, 1:-1,   -1, 0] = sd.values_p[0, 1, :,-2, 0]           # right
                elif iface == 1:
                    padded_array[0, -1,    0, 1:-1, 0] = sd.values_p[0,-2, 1, :, 0]          # top
                    padded_array[0, -1,   -1, 1:-1, 0] = sd.values_p[0,-2,-2, :, 0]          # bottom
                    padded_array[0, -1, 1:-1,    0, 0] = sd.values_p[0,-2, :, 1, 0]          # left
                    padded_array[0, -1, 1:-1,   -1, 0] = sd.values_p[0,-2, :,-2, 0]          # right
                elif iface == 2:
                    padded_array[0, 1:-1,    0, 0, 0] = sd.values_p[0,  :,  1, 1, 0]         # top
                    padded_array[0, 1:-1,   -1, 0, 0] = sd.values_p[0,  :, -2, 1, 0]         # bottom
                    padded_array[0,   -1, 1:-1, 0, 0] = sd.values_p[0, -2,  :, 1, 0]         # left
                    padded_array[0,    0, 1:-1, 0, 0] = sd.values_p[0,  1,  :, 1, 0]         # right
                elif iface == 3:
                    padded_array[0, 1:-1,    0, -1, 0] = sd.values_p[0,  :,  1, -2, 0]       # top
                    padded_array[0, 1:-1,   -1, -1, 0] = sd.values_p[0,  :, -2, -2, 0]       # bottom
                    padded_array[0,   -1, 1:-1, -1, 0] = sd.values_p[0, -2,  :, -2, 0]       # left
                    padded_array[0,    0, 1:-1, -1, 0] = sd.values_p[0,  1,  :, -2, 0]       # right
                elif iface == 4:
                    padded_array[0,    -1, 0, 1:-1, 0] = sd.values_p[0, -2, 1,  :, 0]        # top
                    padded_array[0,     0, 0, 1:-1, 0] = sd.values_p[0,  1, 1,  :, 0]        # bottom
                    padded_array[0,  1:-1, 0,    0, 0] = sd.values_p[0,  :, 1,  1, 0]        # left
                    padded_array[0,  1:-1, 0,   -1, 0] = sd.values_p[0,  :, 1, -2, 0]        # right
                elif iface == 5:
                    padded_array[0,   -1, -1, 1:-1, 0] = sd.values_p[0, -2, -2,  :, 0]       # top
                    padded_array[0,    0, -1, 1:-1, 0] = sd.values_p[0,  1, -2,  :, 0]       # bottom
                    padded_array[0, 1:-1, -1,    0, 0] = sd.values_p[0,  :, -2,  1, 0]       # left
                    padded_array[0, 1:-1, -1,   -1, 0] = sd.values_p[0,  :, -2, -2, 0]       # right

            else: # means it has a neighbouring subdomain
                if iface == 0:
                    padded_array[0, 0,    0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_p[0, -1, 0, :, 0])         # top
                    padded_array[0, 0,   -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_p[0, -1,-1, :, 0])         # bottom
                    padded_array[0, 0, 1:-1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_p[0, -1, :, 0, 0])         # left
                    padded_array[0, 0, 1:-1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[0].values_p[0, -1, :,-1, 0])         # right
                elif iface == 1:
                    padded_array[0, -1,    0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_p[0,0, 0, :, 0])          # top
                    padded_array[0, -1,   -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_p[0,0,-1, :, 0])          # bottom
                    padded_array[0, -1, 1:-1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_p[0,0, :, 0, 0])          # left
                    padded_array[0, -1, 1:-1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[1].values_p[0,0, :,-1, 0])          # right
                elif iface == 2:
                    padded_array[0, 1:-1,    0, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_p[0,  :,  0, -1, 0])       # top
                    padded_array[0, 1:-1,   -1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_p[0,  :, -1, -1, 0])       # bottom
                    padded_array[0,   -1, 1:-1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_p[0, -1,  :, -1, 0])       # left
                    padded_array[0,    0, 1:-1, 0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[2].values_p[0,  0,  :, -1, 0])       # right
                elif iface == 3:
                    padded_array[0, 1:-1,    0, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_p[0,  :, 0,0,0])          # top
                    padded_array[0, 1:-1,   -1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_p[0,  :,-1,0,0])          # bottom
                    padded_array[0,   -1, 1:-1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_p[0, -1, :,0,0])          # left
                    padded_array[0,    0, 1:-1, -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[3].values_p[0,  0, :,0,0])          # right
                elif iface == 4:
                    padded_array[0,    -1, 0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_p[0, -1, -1,  :, 0])      # top
                    padded_array[0,     0, 0, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_p[0,  0, -1,  :, 0])      # bottom
                    padded_array[0,  1:-1, 0,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_p[0,  :, -1,  0, 0])      # left
                    padded_array[0,  1:-1, 0,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[4].values_p[0,  :, -1, -1, 0])      # right
                elif iface == 5:
                    padded_array[0,   -1, -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_p[0, -1, 0,  :, 0])       # top
                    padded_array[0,    0, -1, 1:-1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_p[0,  0, 0,  :, 0])       # bottom
                    padded_array[0, 1:-1, -1,    0, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_p[0,  :, 0,  0, 0])       # left
                    padded_array[0, 1:-1, -1,   -1, 0] = scale_arrays(sd.values_u[0, -2, -2,  :, 0], sd.neig[5].values_p[0,  :, 0, -1, 0])       # right

            # corner points
            if sd.corner_neig[0] == 'No':                                                       # node 0 == 6 of corner_neig
                padded_array[0,  0, -1,  0, 0] = sd.values_p[0,  1, -2,  1, 0] 
            elif not isinstance(sd.corner_neig[0], str):   
                padded_array[0,  0, -1,  0, 0] = sd.corner_neig[0].values_p[0, -2,1,-2, 0]
            
            if sd.corner_neig[1] == 'No':                                                        # node 1 == 7 of corner_neig
                padded_array[0,  0, -1, -1, 0] = sd.values_p[0,  1, -2, -2, 0]
            elif not isinstance(sd.corner_neig[1], str):
                padded_array[0,  0, -1, -1, 0] = sd.corner_neig[1].values_p[0, -2,1,1, 0]

            if sd.corner_neig[2] == 'No':                                                        # node 2 == 4 of corner_neig
                padded_array[0,  0,  0, -1, 0] = sd.values_p[0,  1,  1, -2, 0]
            elif not isinstance(sd.corner_neig[2], str):
                padded_array[0,  0,  0, -1, 0] = sd.corner_neig[2].values_p[0, -2,-2,1, 0]

            if sd.corner_neig[3] == 'No':                                                        # node 3 == 5 of corner_neig
                padded_array[0,  0,  0,  0, 0] = sd.values_p[0,  1,  1,  1, 0]
            elif not isinstance(sd.corner_neig[3], str):
                padded_array[0,  0,  0,  0, 0] = sd.corner_neig[3].values_p[0, -2,-2,-2, 0]

            if sd.corner_neig[4] == 'No':                                                        # node 4 == 2 of corner_neig
                padded_array[0, -1, -1,  0, 0] = sd.values_p[0, -2, -2,  1, 0]
            elif not isinstance(sd.corner_neig[4], str):
                padded_array[0, -1, -1,  0, 0] = sd.corner_neig[4].values_p[0, 1,1,-2, 0]

            if sd.corner_neig[5] == 'No':                                                        # node 5 == 3 of corner_neig
                padded_array[0, -1, -1, -1, 0] = sd.values_p[0, -2, -2, -2, 0]
            elif not isinstance(sd.corner_neig[5], str):
                padded_array[0, -1, -1, -1, 0] = sd.corner_neig[5].values_p[0, 1,1,1, 0]

            if sd.corner_neig[6] == 'No':                                                        # node 6 == 0 of corner_neig
                padded_array[0, -1,  0, -1, 0] = sd.values_p[0, -2,  1, -2, 0]
            elif not isinstance(sd.corner_neig[6], str):
                padded_array[0, -1,  0, -1, 0] = sd.corner_neig[6].values_p[0, 1,-2, 1, 0]

            if sd.corner_neig[7] == 'No':                                                        # node 7 == 1 of corner_neig
                padded_array[0, -1,  0,  0, 0] = sd.values_p[0, -2,  1,  1, 0]
            elif not isinstance(sd.corner_neig[7], str):
                padded_array[0, -1,  0,  0, 0] = sd.corner_neig[7].values_p[0, 1,-2,-2, 0]


    return padded_array



def padd_ml(sd, ilevel):
    nx = int(len(sd.w[ilevel][0,0,0,:,0]))
    padded_array = np.zeros((1,nx + 2, nx + 2,nx+2, 1))
    padded_array[:, 1:-1, 1:-1, 1:-1, :] = sd.w[ilevel]
    
    for iface in [0,1,4,5,2,3]: # check the outlet last to make sure it sets all values to 0
        if sd.neig[iface] == 'outlet':
            padded_array[0, :, :, -1, 0] = 0
            
        else:
            if sd.neig[iface] == 'side' or sd.neig[iface] == 'inlet':
                if iface == 0:
                    padded_array[0,    0, 1:-1, 1:-1, 0] = sd.w[ilevel][0, 1,  :,  :, 0]                        # face
                elif iface == 1:
                    padded_array[0,   -1, 1:-1, 1:-1, 0] = sd.w[ilevel][0,-2,  :,  :, 0]                        # face
                elif iface == 2:
                    padded_array[0, 1:-1, 1:-1,    0, 0] = sd.w[ilevel][0, :,  :,  1, 0]                        # face
                elif iface == 3:
                    padded_array[0, 1:-1, 1:-1,   -1, 0] = sd.w[ilevel][0, :,  :, -2, 0]                        # face
                elif iface == 4:
                    padded_array[0, 1:-1,    0, 1:-1, 0] = sd.w[ilevel][0, :,  1,  :, 0]                        # face
                elif iface == 5:
                    padded_array[0, 1:-1,   -1, 1:-1, 0] = sd.w[ilevel][0, :, -2,  :, 0]                        # face
            
            else: # means it has a neighbouring subdomain
                if iface == 0:
                    padded_array[0,    0, 1:-1, 1:-1, 0] = scale_faces(sd.w[ilevel][0, 0,  :,  :, 0], sd.neig[0].w[ilevel][0, -1,  :,  :, 0])               # face
                elif iface == 1:
                    padded_array[0,   -1, 1:-1, 1:-1, 0] = scale_faces(sd.w[ilevel][0, -1,  :,  :, 0], sd.neig[1].w[ilevel][0,  0,  :,  :, 0])               # face
                elif iface == 2:
                    padded_array[0, 1:-1, 1:-1,    0, 0] = scale_faces(sd.w[ilevel][0, :,  :,  0, 0], sd.neig[2].w[ilevel][0,  :,  :, -1, 0])               # face
                elif iface == 3:
                    padded_array[0, 1:-1, 1:-1,   -1, 0] = scale_faces(sd.w[ilevel][0, :,  :,  -1, 0], sd.neig[3].w[ilevel][0,  :,  :,  0, 0])               # face
                elif iface == 4:
                    padded_array[0, 1:-1,    0, 1:-1, 0] = scale_faces(sd.w[ilevel][0, :,  0,  :, 0], sd.neig[4].w[ilevel][0,  :, -1,  :, 0])               # face
                elif iface == 5:
                    padded_array[0, 1:-1,   -1, 1:-1, 0] = scale_faces(sd.w[ilevel][0, :,  -1,  :, 0], sd.neig[5].w[ilevel][0,  :,  0,  :, 0])               # face
                    
            ''' corner points
            as explained when corner neigs are defined. here we do not consider some corner neig s for computer efficinecy. 
            they may need to be defined for other cases
            '''
            for i in range(12):
                if not isinstance(sd.corner_neig[i], str):
                    if i == 5:
                        padded_array[0,   1:-1,  0,  0, 0] = scale_arrays(sd.w[ilevel][0, :,  0,  0, 0], sd.corner_neig[i].w[ilevel][0,:,-1,-1,0])
                    elif i == 6:
                        padded_array[0,   1:-1,  0, -1, 0] = scale_arrays(sd.w[ilevel][0, :,  0,  0, 0], sd.corner_neig[i].w[ilevel][0,:,-1, 0,0])
                    elif i == 9:
                        padded_array[0,   1:-1, -1,  0, 0] = scale_arrays(sd.w[ilevel][0, :,  0,  0, 0], sd.corner_neig[i].w[ilevel][0,:, 0, 1,0])
                    elif i == 10:
                        padded_array[0,   1:-1, -1, -1, 0] = scale_arrays(sd.w[ilevel][0, :,  0,  0, 0], sd.corner_neig[i].w[ilevel][0,:, 0, 0,0])
        
        
        
            if sd.corner_node_neig[0] == 'No':                                                       # node 0 == 6 of corner_neig
                padded_array[0,  0, -1,  0, 0] = sd.w[ilevel][0,  1, -2,  1, 0] 
            else:   
                padded_array[0,  0, -1,  0, 0] = sd.corner_node_neig[0].w[ilevel][0, -2,1,-2, 0]
            
            if sd.corner_node_neig[1] == 'No':                                                        # node 1 == 7 of corner_neig
                padded_array[0,  0, -1, -1, 0] = sd.w[ilevel][0,  1, -2, -2, 0]
            else:
                padded_array[0,  0, -1, -1, 0] = sd.corner_node_neig[1].w[ilevel][0, -2,1,1, 0]

            if sd.corner_node_neig[2] == 'No':                                                        # node 2 == 4 of corner_neig
                padded_array[0,  0,  0, -1, 0] = sd.w[ilevel][0,  1,  1, -2, 0]
            else:
                padded_array[0,  0,  0, -1, 0] = sd.corner_node_neig[2].w[ilevel][0, -2,-2,1, 0]

            if sd.corner_node_neig[3] == 'No':                                                        # node 3 == 5 of corner_neig
                padded_array[0,  0,  0,  0, 0] = sd.w[ilevel][0,  1,  1,  1, 0]
            else:
                padded_array[0,  0,  0,  0, 0] = sd.corner_node_neig[3].w[ilevel][0, -2,-2,-2, 0]

            if sd.corner_node_neig[4] == 'No':                                                        # node 4 == 2 of corner_neig
                padded_array[0, -1, -1,  0, 0] = sd.w[ilevel][0, -2, -2,  1, 0]
            else:
                padded_array[0, -1, -1,  0, 0] = sd.corner_node_neig[4].w[ilevel][0, 1,1,-2, 0]

            if sd.corner_node_neig[5] == 'No':                                                        # node 5 == 3 of corner_neig
                padded_array[0, -1, -1, -1, 0] = sd.w[ilevel][0, -2, -2, -2, 0]
            else:
                padded_array[0, -1, -1, -1, 0] = sd.corner_node_neig[5].w[ilevel][0, 1,1,1, 0]

            if sd.corner_node_neig[6] == 'No':                                                        # node 6 == 0 of corner_neig
                padded_array[0, -1,  0, -1, 0] = sd.w[ilevel][0, -2,  1, -2, 0]
            else:
                padded_array[0, -1,  0, -1, 0] = sd.corner_node_neig[6].w[ilevel][0, 1,-2, 1, 0]

            if sd.corner_node_neig[7] == 'No':                                                        # node 7 == 1 of corner_neig
                padded_array[0, -1,  0,  0, 0] = sd.w[ilevel][0, -2,  1,  1, 0]
            else:
                padded_array[0, -1,  0,  0, 0] = sd.corner_node_neig[7].w[ilevel][0, 1,-2,-2, 0]
    
    return padded_array


def PG_padding_ku(sd, k_u, dt):
    padded_array = np.zeros((1,sd.nz + 2, sd.ny + 2 , sd.nx + 2,1))
    padded_array[:, 1:-1, 1:-1, 1:-1, :] = k_u
    
    # padded_array[0, 1:-1, 1:-1,  0,0] = np.ones((128,128)) # ------------------------------------------------------------------------------------------------------------ why ?!!!
    for iface in range(2,6):
        if not isinstance(sd.neig[iface], str):
            # iface 0 & 1 is not considered here az nz =1
            if iface == 2:
                padded_array[0, 1:-1, 1:-1,  0,0]  = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_u(sd.neig[iface], dt)[0, :, :, -1,0])
            elif iface == 3:
                padded_array[0, 1:-1, 1:-1, -1,0]  = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_u(sd.neig[iface], dt)[0, :, :, 0,0]) 
            elif iface == 4:
                padded_array[0, 1:-1,  0, 1:-1,0]  = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_u(sd.neig[iface], dt)[0, :, -1, :,0])
            elif iface == 5:
                padded_array[0, 1:-1, -1, 1:-1,0]  = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_u(sd.neig[iface], dt)[0, :, 0, :,0])
    
    return padded_array


def PG_padding_kv(sd, k_v, dt):
    padded_array = np.zeros((1,sd.nz + 2, sd.ny + 2 , sd.nx + 2,1))
    padded_array[:, 1:-1, 1:-1, 1:-1, :] = k_v
    
    for iface in range(2,6):
        if not isinstance(sd.neig[iface], str):
            # iface 0 & 1 is not considered here az nz =1
            if iface == 2:
                padded_array[0,1:-1,1:-1,0,0]  = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_v(sd.neig[iface], dt)[0,:,:,-1,0])
            elif iface == 3:
                padded_array[0,1:-1,1:-1,-1,0] = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_v(sd.neig[iface], dt)[0,:,:,0,0])
            elif iface == 4:
                padded_array[0,1:-1,0,1:-1,0]  = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_v(sd.neig[iface], dt)[0,:,-1,:,0])
            elif iface == 5:
                padded_array[0,1:-1,-1,1:-1,0] = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_v(sd.neig[iface], dt)[0,:,0,:,0])

    return padded_array


def PG_padding_kw(sd, k_w, dt):
    padded_array = np.zeros((1,sd.nz + 2, sd.ny + 2 , sd.nx + 2,1))
    padded_array[:, 1:-1, 1:-1, 1:-1, :] = k_w
    
    for iface in range(2,6):
        if not isinstance(sd.neig[iface], str):
            # iface 0 & 1 is not considered here az nz =1
            if iface == 2:
                padded_array[0,1:-1,1:-1,0,0]  = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_w(sd.neig[iface], dt)[0,:,:,-1,0])
            elif iface == 3:
                padded_array[0,1:-1,1:-1,-1,0] = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_w(sd.neig[iface], dt)[0,:,:,0,0])
            elif iface == 4:
                padded_array[0,1:-1,0,1:-1,0]  = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_w(sd.neig[iface], dt)[0,:,-1,:,0])
            elif iface == 5:
                padded_array[0,1:-1,-1,1:-1,0] = scale_faces(sd.values_u[0,:,:,0,0], Half_Petrov_Galerkin_dissipation_w(sd.neig[iface], dt)[0,:,0,:,0])

    return padded_array



def Half_Petrov_Galerkin_dissipation_u(sd, dt):    
    values_uu = add_padding_u(sd, sd.values_u)
    # sd.values_u /= (1+dt*sd.sigma) 
    m_i = sd.dx * sd.dy * sd.dz
    
    k_u_neig = (sd.dx + sd.dy + sd.dz)/36 * abs((1/m_i) * 
                                                 (abs(sd.dx * sd.values_u) + abs(sd.dy * sd.values_v) + abs(sd.dz * sd.values_w)) * sd.CNN3D_dif(values_uu)) /\
                                                 (1e-3 + (abs(sd.CNN3D_xadv(values_uu)) + 
                                                          abs(sd.CNN3D_yadv(values_uu)) + 
                                                          abs(sd.CNN3D_zadv(values_uu)))/3 / m_i)
    
    k_u_neig = np.minimum(k_u_neig, np.ones((1,sd.nz, sd.ny,sd.nx,1))*(0.25 * min(sd.dx, sd.dx, sd.dx)**2 /dt)) #/ (1+dt*sd.sigma) 

    return k_u_neig


def Half_Petrov_Galerkin_dissipation_v(sd, dt):    
    values_vv = add_padding_v(sd, sd.values_v)
    # sd.values_v /= (1+dt*sd.sigma)     
    m_i = sd.dx * sd.dy * sd.dz
    
    k_v_neig = (sd.dx + sd.dy + sd.dz)/36 * abs((1/m_i) * (abs(sd.dx * sd.values_u) + abs(sd.dy * sd.values_v) + abs(sd.dz * sd.values_w)) * sd.CNN3D_dif(values_vv)) /\
                                                      (1e-3 + (abs(sd.CNN3D_xadv(values_vv)) + 
                                                               abs(sd.CNN3D_yadv(values_vv)) + 
                                                               abs(sd.CNN3D_zadv(values_vv)))/3 / m_i)
                
    k_v_neig = np.minimum(k_v_neig, np.ones((1,sd.nz, sd.ny,sd.nx,1))*0.25 * min(sd.dx, sd.dx, sd.dx)**2 /dt) #/ (1+dt*sd.sigma)     

    return k_v_neig


def Half_Petrov_Galerkin_dissipation_w(sd, dt):    
    values_ww = add_padding_w(sd, sd.values_w) 
    # sd.values_w /= (1+dt*sd.sigma)
    m_i = sd.dx * sd.dy * sd.dz
    
    k_w_neig = (sd.dx + sd.dy + sd.dz)/36 * abs((1/m_i) * (abs(sd.dx * sd.values_u) + abs(sd.dy * sd.values_v) + abs(sd.dz * sd.values_w)) * sd.CNN3D_dif(values_ww)) /\
                                                     (1e-3 + (abs(sd.CNN3D_xadv(values_ww)) + 
                                                              abs(sd.CNN3D_yadv(values_ww)) + 
                                                              abs(sd.CNN3D_zadv(values_ww)))/3 / m_i)
    
    k_w_neig = np.minimum(k_w_neig, np.ones((1,sd.nz, sd.ny,sd.nx,1))*0.25 * min(sd.dx, sd.dx, sd.dx)**2 /dt) #/ (1+dt*sd.sigma) 

    return k_w_neig


def Petrov_Galerkin_dissipation(sd, CNN3D_dif, CNN3D_xadv, CNN3D_yadv, CNN3D_zadv, values_u, values_v, values_w, values_uu, values_vv, values_ww,dt):    
    '''Turbulence modelling using Petrov-Galerkin dissipation       
    eplsion_k: Need to sufficiently large
    sigma: is a tensor which defines the location of the buildings
    Output
    '''
    # values_u /= (1+dt*sd.sigma) 
    # values_v /= (1+dt*sd.sigma)     
    # values_w /= (1+dt*sd.sigma)
    # values_uu = values_uu / (1+dt*sd.pad_sigma) 
    # values_vv = values_vv / (1+dt*sd.pad_sigma)     
    # values_ww = values_ww / (1+dt*sd.pad_sigma)
    
    
    # numerator = sd.pg_cst * (abs(sd.dx * sd.values_u) + abs(sd.dy * sd.values_v) + abs(sd.dz * sd.values_w))
    
    # k_u = abs( numerator * CNN3D_dif(values_uu)) / (1e-3 + (abs(CNN3D_xadv(values_uu)) + 
    #                                                         abs(CNN3D_yadv(values_uu)) + 
    #                                                         abs(CNN3D_zadv(values_uu)))/3 / sd.m_i)
    
    # k_v = abs(numerator * CNN3D_dif(values_vv)) / (1e-3 + ( abs(CNN3D_xadv(values_vv)) + 
    #                                                         abs(CNN3D_yadv(values_vv)) + 
    #                                                         abs(CNN3D_zadv(values_vv)))/3 / sd.m_i)
                
    # k_w = abs(numerator * CNN3D_dif(values_ww)) / (1e-3 + ( abs(CNN3D_xadv(values_ww)) + 
    #                                                         abs(CNN3D_yadv(values_ww)) + 
    #                                                         abs(CNN3D_zadv(values_ww)))/3 / sd.m_i)
    
    k_i = np.minimum(sd.k_u, sd.kmax) #/ (1+dt*sd.sigma) 
    k_j = np.minimum(sd.k_v, sd.kmax) #/ (1+dt*sd.sigma)     
    k_k = np.minimum(sd.k_w, sd.kmax) #/ (1+dt*sd.sigma) 

    # k_u = np.ones(input_shape)*0.05  / (1+dt*sd.sigma) 
    # k_v = np.ones(input_shape)*0.05  / (1+dt*sd.sigma)     
    # k_w = np.ones(input_shape)*0.05  / (1+dt*sd.sigma)   

    # Amin:: if you must include division by sigma, just use pad_sigma as the normal sd.sigma gives NaN
    k_uu = PG_padding_ku(sd, k_i, dt) #/ (1+dt*sd.pad_sigma)
    k_vv = PG_padding_kv(sd, k_j, dt) #/ (1+dt*sd.pad_sigma)
    k_ww = PG_padding_kw(sd, k_k, dt) #/ (1+dt*sd.pad_sigma)

    k_x = 0.5*(k_i*CNN3D_dif(values_uu) +
                   CNN3D_dif(values_uu*k_uu) -
          values_u*CNN3D_dif(k_uu))


    k_y = 0.5*(k_j*CNN3D_dif(values_vv) + 
                   CNN3D_dif(values_vv*k_vv) -
          values_v*CNN3D_dif(k_vv))


    k_z = 0.5*(k_k*CNN3D_dif(values_ww) + 
                   CNN3D_dif(values_ww*k_ww) -
          values_w*CNN3D_dif(k_ww))
    
    return 8*k_x, 8*k_y, 8*k_z





def solve_velocity_step_one(sd, CNN3D_dif, CNN3D_xadv, CNN3D_yadv, CNN3D_zadv, DPG,dt, Re):
    values_uu = add_padding_u(sd, sd.values_u)
    values_vv = add_padding_v(sd, sd.values_v)
    values_ww = add_padding_w(sd, sd.values_w) 

    # ------------------------------------------------- apply Petrov Galerkin if needed
    if DPG:
        [k_x,k_v,k_w] = Petrov_Galerkin_dissipation(sd, CNN3D_dif, CNN3D_xadv, CNN3D_yadv, CNN3D_zadv, sd.values_u, sd.values_v, sd.values_w, values_uu, values_vv, values_ww, dt)
    else:
        k_x = CNN3D_dif(values_uu)
        k_v = CNN3D_dif(values_vv)
        k_w = CNN3D_dif(values_ww)
    # -------------------------------------------------
    a_u = dt*Re/sd.dx**2 * k_x.numpy() - \
          sd.values_u*CNN3D_xadv(values_uu).numpy() - \
          sd.values_v*CNN3D_yadv(values_uu).numpy() - \
          sd.values_w*CNN3D_zadv(values_uu).numpy()
    
    a_v = dt*Re/sd.dy**2 * k_v.numpy() - \
          sd.values_u*CNN3D_xadv(values_vv).numpy() - \
          sd.values_v*CNN3D_yadv(values_vv).numpy() - \
          sd.values_w*CNN3D_zadv(values_vv).numpy()
    
    a_w = dt*Re/sd.dz**2 * k_w.numpy() - \
          sd.values_u*CNN3D_xadv(values_ww).numpy() - \
          sd.values_v*CNN3D_yadv(values_ww).numpy() - \
          sd.values_w*CNN3D_zadv(values_ww).numpy()
    
    b_u = 0.5*a_u + sd.values_u
    b_v = 0.5*a_v + sd.values_v
    b_w = 0.5*a_w + sd.values_w
    
    return b_u, b_v, b_w


def solve_velocity_step_two(sd, b_u, b_v, b_w, CNN3D_dif, CNN3D_xadv, CNN3D_yadv, CNN3D_zadv, DPG,dt, Re):
    values_uu = add_padding_u(sd, b_u)
    values_vv = add_padding_v(sd, b_v)
    values_ww = add_padding_w(sd, b_w)

    # ------------------------------------------------- apply Petrov Galerkin if needed
    if DPG:
        [k_x,k_v,k_w] = Petrov_Galerkin_dissipation(sd, CNN3D_dif, CNN3D_xadv, CNN3D_yadv, CNN3D_zadv, sd.values_u, sd.values_v, sd.values_w, values_uu, values_vv, values_ww,dt)
    else:
        k_x = CNN3D_dif(values_uu)
        k_v = CNN3D_dif(values_vv)
        k_w = CNN3D_dif(values_ww)
    # -------------------------------------------------
    c_u = dt*Re/sd.dx**2 * k_x.numpy() - \
          b_u*CNN3D_xadv(values_uu).numpy() - \
          b_v*CNN3D_yadv(values_uu).numpy() - \
          b_w*CNN3D_zadv(values_uu).numpy()
    
    c_v = dt*Re/sd.dy**2 * k_v.numpy() - \
          b_u*CNN3D_xadv(values_vv).numpy() - \
          b_v*CNN3D_yadv(values_vv).numpy() - \
          b_w*CNN3D_zadv(values_vv).numpy()
    
    c_w = dt*Re/sd.dz**2 * k_w.numpy() - \
          b_u*CNN3D_xadv(values_ww).numpy() - \
          b_v*CNN3D_yadv(values_ww).numpy() - \
          b_w*CNN3D_zadv(values_ww).numpy()
    
    sd.values_u = sd.values_u + c_u
    sd.values_v = sd.values_v + c_v
    sd.values_w = sd.values_w + c_w



# def get_momentum(domain_info, DPG, dt, Re):
#     '''
#     solves momentum equations for all subdomains except sd0 which has the bluff body based on subdomain information passed as a list as below:
#     [0= sd, 1= CNN3D_dif, 2= CNN3D_xadv, 3= CNN3D_yadv, 4= CNN3D_zadv]
#     '''
#     no_domains = len(domain_info)

#     for i in range(no_domains):
#         # update boundaries
#         # physical_halos_vel(domain_info[i][0], domain_info[i][0].values_u, domain_info[i][0].values_v, domain_info[i][0].values_w)
#         # ------------ Two-stepping scheme for advection and diffusion  Eq 27 ----------------      
#         b_u , b_v, b_w = solve_velocity_step_one(domain_info[i][0], domain_info[i][1], domain_info[i][2], domain_info[i][3], domain_info[i][4], DPG,dt, Re)

#     for i in range(no_domains):    
#         physical_halos_vel(domain_info[i][0],b_u, b_v, b_w)
#         update_inner_halo_vel(domain_info[i][0])

#     for i in range(no_domains):
#         solve_velocity_step_two(domain_info[i][0], b_u, b_v, b_w, domain_info[i][1], domain_info[i][2], domain_info[i][3], domain_info[i][4], DPG,dt,Re)


def get_momentum(sd, DPG, dt, Re):
    '''
    solves momentum equations for all subdomains except sd0 which has the bluff body based on subdomain information passed as a list as below:
    [0= sd]
    '''
    no_domains = len(sd)

    for i in range(no_domains):

        if DPG:
            values_uu = add_padding_u(sd[i], sd[i].values_u)
            values_vv = add_padding_v(sd[i], sd[i].values_v)
            values_ww = add_padding_w(sd[i], sd[i].values_w) 
            numerator = sd[i].pg_cst * (abs(sd[i].dx * sd[i].values_u) + abs(sd[i].dy * sd[i].values_v) + abs(sd[i].dz * sd[i].values_w))
        
            sd[i].k_u = abs( numerator * sd[i].CNN3D_dif(values_uu)) / (1e-3 + (abs(sd[i].CNN3D_xadv(values_uu)) + 
                                                                                abs(sd[i].CNN3D_yadv(values_uu)) + 
                                                                                abs(sd[i].CNN3D_zadv(values_uu)))/3 / sd[i].m_i)
            
            sd[i].k_v = abs(numerator * sd[i].CNN3D_dif(values_vv)) / (1e-3 + ( abs(sd[i].CNN3D_xadv(values_vv)) + 
                                                                                abs(sd[i].CNN3D_yadv(values_vv)) + 
                                                                                abs(sd[i].CNN3D_zadv(values_vv)))/3 / sd[i].m_i)
                        
            sd[i].k_w = abs(numerator * sd[i].CNN3D_dif(values_ww)) / (1e-3 + ( abs(sd[i].CNN3D_xadv(values_ww)) + 
                                                                                abs(sd[i].CNN3D_yadv(values_ww)) + 
                                                                                abs(sd[i].CNN3D_zadv(values_ww)))/3 / sd[i].m_i)

    for i in range(no_domains):
        # update boundaries
        # physical_halos_vel(sd[i], sd[i].values_u, sd[i].values_v, sd[i].values_w)
        # update_inner_halo_vel(sd[i])
            
        b_u , b_v, b_w = solve_velocity_step_one(sd[i], sd[i].CNN3D_dif, sd[i].CNN3D_xadv, sd[i].CNN3D_yadv, sd[i].CNN3D_zadv, DPG,dt,Re)
    
        # physical_halos_vel(sd[i],b_u, b_v, b_w)
        # update_inner_halo_vel(sd[i])

        solve_velocity_step_two(sd[i], b_u, b_v, b_w,  sd[i].CNN3D_dif, sd[i].CNN3D_xadv, sd[i].CNN3D_yadv, sd[i].CNN3D_zadv, DPG,dt, Re)
        
        # physical_halos_vel(sd[i], sd[i].values_u, sd[i].values_v, sd[i].values_w)
        # update_inner_halo_vel(sd[i])
    


def do_multigrid(sd , multi_itr, j_itr, nlevel, no_domains):
    '''0= sd0, 1= w_A_sd0, 2= CNN3D_A_padd0, 3= CNN3D_res_sd0, 4= CNN3D_prol_sd0]'''
    
    for multi_grid in range(multi_itr):
    # ----------------------------------------------------------------------- residuale
        #create empty list to store residuales of each subdomain
        res_list = [[] for _ in range(no_domains)]


        # ---------------------------------------------------------------------------------------------------------- coarsest w
        for i in range(no_domains):
            k = np.shape( sd[i].w[nlevel][0,0,:,0,0])[0]
            sd[i].w[nlevel] = np.zeros([1,k,k,k,1])

        for i in range(no_domains):
            # only the finest grid level residual
            res_list[i].append(sd[i].CNN3D_A_padd[0](add_padding_p(sd[i])).numpy() - sd[i].b)

            # residuale for the rest of the grid levels
            for j in range(sd[i].nlevel-2, sd[i].nlevel-2 - nlevel, -1):
                res_list[i].append(sd[i].CNN3D_res[j](res_list[i][-1]))  

        # ---------------------------------------------------------------------------------------------------------- jacobi
        for ilevel in range(nlevel,0,-1):
            for jacobi in range(j_itr):
                for i in range(no_domains):
                    sd[i].w[ilevel] = (sd[i].w[ilevel] - sd[i].CNN3D_A_padd[ilevel](padd_ml(sd[i],ilevel)) /
                                       sd[i].w_A[0,1,1,1,0] + res_list[i][ilevel] /sd[i].w_A[0,1,1,1,0]).numpy()

            # ----------------------------------------------------------------------- prolong for other sd
            for i in range(no_domains):
                sd[i].w[ilevel-1] = sd[i].CNN3D_prol[-ilevel](sd[i].w[ilevel]).numpy()

            # ---------------------------------------------------------------------------------------------------------- correct pressure
        for i in range(no_domains):
            sd[i].values_p = sd[i].values_p - sd[i].w[0]

            # ---------------------------------------------------------------------------------------------------------- update inner halos
        for i in range(no_domains):
            update_inner_halo_p(sd[i])
            read_physical_halos_p(sd[i])

            # ---------------------------------------------------------------------------------------------------------- solve for the finest grid level
        for i in range(no_domains):
            sd[i].values_p = (sd[i].values_p - sd[i].CNN3D_A_padd[0](add_padding_p(sd[i])).numpy()/
                                               sd[i].w_A[0,1,1,1,0] + sd[i].b/sd[i].w_A[0,1,1,1,0])

            # ---------------------------------------------------------------------------------------------------------- update inner halos
        for i in range(no_domains):
            update_inner_halo_p(sd[i])
            read_physical_halos_p(sd[i])



    
def grad_p(sd):
    '''
    solves grad P based on subdomain information passed as a list as below:
    [0= sd1]
    '''
    no_domains = len(sd)

    for i in range(no_domains):
        padded_p = add_padding_p(sd[i])
        sd[i].values_u = sd[i].values_u - sd[i].CNN3D_pu(padded_p).numpy()
        sd[i].values_v = sd[i].values_v - sd[i].CNN3D_pv(padded_p).numpy()
        sd[i].values_w = sd[i].values_w - sd[i].CNN3D_pw(padded_p).numpy() 

        # physical_halos_vel( sd[i], sd[i].values_u, sd[i].values_v, sd[i].values_w)
        # update_inner_halo_vel(sd[i])

    # for i in range(no_domains):
    #     physical_halos_vel( sd[i], sd[i].values_u, sd[i].values_v, sd[i].values_w)
        # update_inner_halo_vel(sd[i])


def get_b(sd):
    '''
    now we need to sol S in the diagram
    It calculates 'b' based on domain_info list. follow below for the inputs
    0= sd, 1= CNN2D_Su_sd, 2= CNN2D_Sv_sd, 3= CNN2D_Sw_sd
    '''
    no_domains = len(sd)

    for i in range(no_domains):
        sd[i][0].b = -( sd[i][1](add_padding_u( sd[i][0], sd[i][0].values_u)).numpy() +
                        sd[i][2](add_padding_v( sd[i][0], sd[i][0].values_v)).numpy() +
                        sd[i][3](add_padding_w( sd[i][0], sd[i][0].values_w)).numpy()) 
        














