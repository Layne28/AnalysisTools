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
        corr, corr_sparse, noise_mean, noise_var, quantity_mean, quantity_var, quantity_mean_sparse, quantity_var_sparse = correlate_density(particle_traj, noise_traj, noise_file)
        #### Output field properties to file in same directory as input file ####        
        outfile = '/'.join((particle_file.split('/'))[:-1]) + '/%s_noise_correlation.npz' % quantity
        np.savez(outfile, one_point_corr=corr, one_point_corr_norm=corr/np.sqrt(noise_var*quantity_var), one_point_corr_sparse=corr_sparse, one_point_corr_sparse_norm=corr_sparse/np.sqrt(noise_var*quantity_var_sparse), mean_density=quantity_mean, mean_density_sparse=quantity_mean_sparse, var_density=quantity_var, var_density_sparse=quantity_var_sparse, mean_noise=noise_mean, var_noise=noise_var)
    elif quantity=='pressure':
        corr, corr_abs, noise_mean, noise_var, quantity_mean, quantity_var, quantity_mean_abs, quantity_var_abs = correlate_pressure(particle_traj)
        #### Output field properties to file in same directory as input file ####        
        outfile = '/'.join((particle_file.split('/'))[:-1]) + '/%s_noise_correlation.npz' % quantity
        np.savez(outfile, one_point_corr=corr, one_point_corr_abs=corr_abs, one_point_corr_norm=corr/np.sqrt(noise_var*quantity_var), one_point_corr_abs_norm=corr_abs/np.sqrt(noise_var*quantity_var_abs), mean_pressure=quantity_mean, var_pressure=quantity_var, mean_pressure_abs=quantity_mean_abs, var_pressure_abs=quantity_var_abs, mean_noise=noise_mean, var_noise=noise_var)
    else:
        print('Error: quantity not yet supported.')
        exit()

    

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
    if noise_traj['noise'].shape[0]==1:
        noise_field = np.repeat(noise_traj['noise'], repeats=nframes-nskip, axis=0)
    else:
        noise_field = noise_traj['noise'][nskip:,:,:,:]
    print('noise field', noise_field.shape)
    mag_field = np.sqrt(noise_field[:,:,:,0].transpose(0,2,1)**2+noise_field[:,:,:,1].transpose(0,2,1)**2)
    print('mag field', mag_field.shape)

    #Get density field
    density_field = get_density_field(nframes, nskip, particle_traj['pos'], edges, mag_field, spacing, dims)
    density_field_sparse = get_density_field(nframes, nskip, particle_traj['pos'], edges, mag_field, spacing, dims, do_sparse=1)
    if do_print_density==1:
        np.savez('/'.join((noise_file.split('/'))[:-1]) + '/density_traj.npz', density=density_field)

    #Get noise-density correlation
    noise_mean = np.average(mag_field)
    noise_var = np.var(mag_field)
    density_mean = np.average(density_field)
    density_var = np.var(density_field)
    density_mean_sparse = np.average(density_field_sparse)
    density_var_sparse = np.var(density_field_sparse)

    corr = np.average(np.multiply(mag_field, density_field)) - noise_mean*density_mean
    corr_sparse = np.average(np.multiply(mag_field, density_field_sparse)) - noise_mean*density_mean_sparse

    return corr, corr_sparse, noise_mean, noise_var, density_mean, density_var, density_mean_sparse, density_var_sparse

def correlate_pressure(particle_traj):

    #TODO: make this work for 3d

    nframes = particle_traj['pos'].shape[0]
    nskip = int(nframes*0.4)

    virial = particle_traj['virial'][nskip:,...]
    pressure = -(virial[:,:,0]+virial[:,:,3])/2.0
    active_force = particle_traj['active_force'][nskip:,...]
    active_mag = np.sqrt(active_force[...,0]**2+active_force[...,1]**2)
    print(pressure.shape)
    print(active_mag.shape)

    noise_mean = np.average(active_mag)
    noise_var = np.var(active_mag)
    pressure_mean = np.average(pressure)
    pressure_var = np.var(pressure)
    pressure_abs_mean = np.average(np.abs(pressure))
    pressure_abs_var = np.var(np.abs(pressure))

    corr = np.average(np.multiply(active_mag, pressure)) - noise_mean*pressure_mean
    corr_abs = np.average(np.multiply(active_mag, np.abs(pressure))) - noise_mean*pressure_abs_mean

    return corr, corr_abs, noise_mean, noise_var, pressure_mean, pressure_var, pressure_abs_mean, pressure_abs_var

@numba.jit(nopython=True)
def get_density_field(nframes, nskip, pos, edges, mag_field, spacing, dims, do_sparse=0):

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
            if do_sparse==0:
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
