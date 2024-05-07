#Correlate properties of active noise field with density/pressure fields

import numpy as np
import h5py
import sys
import numba

import AnalysisTools.particle_io as io
import AnalysisTools.measurement_tools as tools

#TODO: make this work for dim=3, not just 2
def main():

    ### Load data ####
    particle_file = sys.argv[1]
    noise_file = sys.argv[2]
    quantity = sys.argv[3] #density or pressure 
    
    particle_traj = io.load_traj(particle_file)
    noise_traj = io.load_noise_traj(noise_file)
    if quantity=='density':
        corr = correlate_density(particle_traj, noise_traj, noise_file)
    elif quantity=='pressure':
        corr = correlate_pressure(particle_traj, noise_traj)
    else:
        print('Error: quantity not yet supported.')
        exit()

    #### Output field properties to file in same directory as input file ####        
    outfile = '/'.join((particle_file.split('/'))[:-1]) + '/%s_noise_correlation.npz' % property
    np.savez(outfile, one_point_corr=corr)

def correlate_density(particle_traj, noise_traj, noise_file):

    do_print_density=0
    #Compute scaled positions
    edges = particle_traj['edges']
    spacing = noise_traj['spacing']
    dims = noise_traj['ncells']
    #Skip first 40% of trajectory (consider it as equilibration)
    nframes = particle_traj['pos'].shape[0]
    nskip = int(nframes*0.4)

    #Get magnitude of noise field
    noise_field = noise_traj['noise'][nskip:,:,:,:]
    mag_field = np.sqrt(noise_field[:,:,:,0].transpose(0,2,1)**2+noise_field[:,:,:,1].transpose(0,2,1)**2)

    #Get density field
    density_field = get_density_field(nframes, nskip, particle_traj['pos'], edges, mag_field, spacing, dims)
    if do_print_density==1:
        np.savez('/'.join((noise_file.split('/'))[:-1]) + '/density_traj.npz', density=density_field)

    #Get noise-density correlation
    noise_mean = np.average(mag_field)
    print(noise_mean)
    density_mean = np.average(density_field)
    print(density_mean)
    corr = np.average(np.multiply(mag_field, density_field)) - noise_mean*density_mean
    print(corr)

    return corr

def correlate_pressure(particle_traj, noise_traj):
    return 0

@numba.jit(nopython=True)
def get_density_field(nframes, nskip, pos, edges, mag_field, spacing, dims):

    halfdiam = 0.5#*2**(1.0/6.0) #half the hard sphere diameter
    print(nframes-nskip)
    pos = pos[nskip:,...]
    scaled_pos_x = (pos[:,:,0]+edges[0]/2.0)/spacing[0]
    scaled_pos_y = (pos[:,:,1]+edges[1]/2.0)/spacing[1]
    scaled_pos_x_int = ((pos[:,:,0]+edges[0]/2.0)/spacing[0]).astype(np.int_)
    scaled_pos_y_int = ((pos[:,:,1]+edges[1]/2.0)/spacing[1]).astype(np.int_)
    print(scaled_pos_x.shape)
    print(mag_field.shape)
    density_field = np.zeros(mag_field.shape)
    for t in range(nframes-nskip):
        for i in range(pos.shape[1]):
            density_field[t,scaled_pos_x_int[t,i],scaled_pos_y_int[t,i]] = 1.0
            #Fill in adjacent sites that lie within the hard sphere diameter
            #scan over a small area adjacent to the central site
            xminind=(scaled_pos_x_int[t,i]-5+dims[0]) % dims[0]
            xmaxind=(scaled_pos_x_int[t,i]+5) % dims[0]
            yminind=(scaled_pos_y_int[t,i]-5+dims[1]) % dims[1]
            ymaxind=(scaled_pos_y_int[t,i]+5) % dims[1]
            for mx in range(scaled_pos_x_int[t,i]-5, scaled_pos_x_int[t,i]+5+1):
                for my in range(scaled_pos_y_int[t,i]-5, scaled_pos_y_int[t,i]+5+1):
                    xind = mx
                    yind = my
                    if xind>=dims[0]:
                        xind -= dims[0]
                    if xind<0:
                        xind += dims[0]
                    if yind>=dims[1]:
                        yind -= dims[1]
                    if yind<0:
                        yind += dims[1]
                    x = xind*spacing[0]+spacing[0]/2.0-edges[0]/2.0
                    y = yind*spacing[1]+spacing[1]/2.0-edges[1]/2.0
                    dx = x-pos[t,i,0]
                    dy = y-pos[t,i,1]
                    if dx>edges[0]/2.0:
                        dx -= edges[0]
                    if dx<=-edges[0]/2.0:
                        dx += edges[0]
                    if dy>edges[1]/2.0:
                        dy -= edges[1]
                    if dy<=-edges[1]/2.0:
                        dy += edges[1]
                    dist = np.sqrt(dx**2+dy**2)
                    if dist<halfdiam:
                        density_field[t,xind,yind] = 1.0

    return density_field

if __name__ == '__main__':
    main()
