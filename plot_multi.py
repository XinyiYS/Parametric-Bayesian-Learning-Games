import numpy as np
import matplotlib.pyplot as plt

# print(plt.style.available)
plt.style.use('seaborn')

LABEL_FONTSIZE = 24
MARKER_SIZE = 10
AXIS_FONTSIZE = 26
TITLE_FONTSIZE= 26
LINEWIDTH = 5

# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('figure', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
plt.rc('axes', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXIS_FONTSIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LABEL_FONTSIZE)    # legend fontsize
plt.rc('lines', markersize=MARKER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=LINEWIDTH)  # fontsize of the figure title


linestyles = ['solid', 'dashed', 'dotted', 'dashdot',]
markers = ['', 'o', 'X', 'P','^', '<', '>' , 'x']


import os
from os.path import join as oj

# result_dir = oj('multiplayer' ,'CaliH', 'P1-1000_500-P2ratio-0.05-0.4')

from contextlib import contextmanager

@contextmanager
def cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)



def plot_SV(exp_dir):

    # result_dir = oj('multiplayer', 'xgpe9', 'multiplayer', 'MNIST_VAE', 'P1-size-5000_P2-size-5000_P1-ratio-0.6')
    for result_dir in os.listdir():
        try:
            with cwd(result_dir):

                N = 3 if 'synthetic' in exp_dir else 4

                player_sample_size_lists = [np.loadtxt('cumulative_{}.txt'.format(i)) for i in range(1 , 1+N) ]
                player_shapley_lists = [np.loadtxt('shapley_fair_{}.txt'.format(i)) for i in range(1 , 1+N) ]
                player_FI_lists = [np.loadtxt('FI_det_{}.txt'.format(i)) for i in range(1 , 1+N) ]

                plt.figure(figsize=(6, 4))

                # Plot sample sizes

                for player_index in range(N):
                    plt.plot(player_sample_size_lists[player_index], label='P'+str(player_index+1), linestyle=linestyles[player_index])
                    
                plt.ylabel('Cumulative count')
                plt.xlabel('Iterations')
                plt.legend()
                plt.tight_layout()
                plt.savefig('output_sharing_rate.png',  bbox_inches='tight')
                # plt.show()
                plt.clf()    
                plt.close()

                plt.figure(figsize=(6, 4))

                # Plot the shapley value
                for player_index in range(N):

                    if 'synthetic' in exp_dir and len(player_shapley_lists[0].shape) > 1:
                        player_shapley_lists = [player_shapley_list[:,0] for player_shapley_list in player_shapley_lists]


                    plt.plot(player_shapley_lists[player_index], label='P'+str(player_index+1), linestyle=linestyles[player_index])
                    # else:
                        # plt.plot(player_shapley_lists[player_index], label='P'+str(player_index+1), linestyle=linestyles[player_index])

                plt.ylabel('Shapley value')
                plt.xlabel('Iterations')

                if 'MNIST_VAE' in exp_dir:
                    bottom, top = plt.ylim() 
                    plt.ylim(top=max(top, 12))

                plt.legend()
                plt.tight_layout()
                plt.savefig('SV.png',  bbox_inches='tight')
                # plt.show()
                plt.clf()    
                plt.close()


                plt.figure(figsize=(6, 4))

                # Plot the FI dets

                for player_index in range(N):
                    plt.plot(player_FI_lists[player_index], label='P'+str(player_index+1), linestyle=linestyles[player_index])

                plt.ylabel('FI Determinant')
                plt.xlabel('Iterations')
                plt.legend()
                plt.tight_layout()
                plt.savefig('fi_determinant.png',  bbox_inches='tight')
                # plt.show()
                plt.clf()    
                plt.close()

        except Exception as e:
            print(result_dir)
            print(e)


def plot_BV(exp_dir):
    # result_dir = oj('multiplayer', 'xgpe9', 'multiplayer', 'MNIST_VAE', 'P1-size-5000_P2-size-5000_P1-ratio-0.6')
    for result_dir in os.listdir():
        
        try:
            with cwd(result_dir):

                N = 3 if 'synthetic' in exp_dir else 4


                player_sample_size_lists = [np.loadtxt('cumulative_{}.txt'.format(i)) for i in range(1 , 1+N) ]
                player_banzhaff_lists = [np.loadtxt('banzhaff_fair_{}.txt'.format(i)) for i in range(1 , 1+N) ]
                player_FI_lists = [np.loadtxt('FI_det_{}.txt'.format(i)) for i in range(1 , 1+N) ]

                plt.figure(figsize=(6, 4))

                # Plot sample sizes

                for player_index in range(N):
                    plt.plot(player_sample_size_lists[player_index], label='P'+str(player_index+1), linestyle=linestyles[player_index])
                    
                plt.ylabel('Cumulative count')
                plt.xlabel('Iterations')
                plt.legend()
                plt.tight_layout()
                plt.savefig('output_sharing_rate.png',  bbox_inches='tight')
                # plt.show()
                plt.clf()    
                plt.close()

                plt.figure(figsize=(6, 4))

                # Plot the banzhaff value
                for player_index in range(N):

                    if 'synthetic' in exp_dir and len(player_banzhaff_lists[0].shape) > 1:
                        player_banzhaff_lists = [player_banzhaff_list[:,0] for player_banzhaff_list in player_banzhaff_lists]


                    plt.plot(player_banzhaff_lists[player_index], label='P'+str(player_index+1), linestyle=linestyles[player_index])
                    # else:
                        # plt.plot(player_banzhaff_lists[player_index], label='P'+str(player_index+1), linestyle=linestyles[player_index])

                plt.ylabel('Banzhaff value')
                plt.xlabel('Iterations')

                if 'MNIST_VAE' in exp_dir:
                    bottom, top = plt.ylim() 
                    plt.ylim(top=max(top, 12))

                plt.legend()
                plt.tight_layout()
                plt.savefig('BV.png',  bbox_inches='tight')
                # plt.show()
                plt.clf()    
                plt.close()


                plt.figure(figsize=(6, 4))

                # Plot the FI dets

                for player_index in range(N):
                    plt.plot(player_FI_lists[player_index], label='P'+str(player_index+1), linestyle=linestyles[player_index])

                plt.ylabel('FI Determinant')
                plt.xlabel('Iterations')
                plt.legend()
                plt.tight_layout()
                plt.savefig('fi_determinant.png',  bbox_inches='tight')
                # plt.show()
                plt.clf()    
                plt.close()

        except Exception as e:
            print(result_dir)
            print(e)





if __name__ == '__main__':
    # exp_dir =  oj('multiplayer', 'xgpe9', 'multiplayer', 'MNIST_VAE')
    exp_dir = oj('multiplayer-BV', 'CaliH')

    try:
        os.chdir(exp_dir)
        plot_BV(exp_dir)

    except:
        pass
