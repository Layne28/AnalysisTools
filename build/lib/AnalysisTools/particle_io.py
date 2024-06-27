#This script contains functions for dealing with h5md trajectory I/O

import h5py
import gsd.hoomd
import gsd.fl
import numpy as np
import sys

def load_traj(myfile):
    
    #Check whether this is an h5 file
    if not(myfile.endswith('.h5') or myfile.endswith('.gsd')):
        print('Error: input to "load_traj" must be h5md or gsd file.')
        return {}

    if myfile.endswith('.h5'):
        traj = h5py.File(myfile)
        traj_dict = {}
        has_topology = 0

        pos = np.array(traj['/particles/all/position/value'])
        vel = np.array(traj['/particles/all/velocity/value'])
        potential_energy = np.array(traj['/observables/potential_energy/value'])
        active_force = np.array(traj['/particles/all/active_force/value'])
        conservative_force = np.array(traj['/particles/all/conservative_force/value'])
        image = np.array(traj['/particles/all/image/value'])
        times = np.array(traj['/particles/all/position/time'])
        edges = np.array(traj['/particles/all/box/edges'])
        dim = traj['/particles/all/box'].attrs['dimension']
        if (('parameters/vmd_structure/bond_from' in traj) and
            ('parameters/vmd_structure/bond_to' in traj)):
            has_topology = 1
            bonds_from = np.array(traj['parameters/vmd_structure/bond_from'])-1
            bonds_to = np.array(traj['parameters/vmd_structure/bond_to'])-1
            bonds = np.vstack((bonds_from, bonds_to)).T
            bonds = np.unique(bonds, axis=0)
        elif ('/particles/all/connectivity' in traj):
            has_topology = 1
            bonds = np.array(traj['/particles/all/connectivity/value'])
        else:
            has_topology = 0


        N = pos.shape[1]
        
        traj.close()

        traj_dict['pos'] = pos
        traj_dict['vel'] = vel
        traj_dict['active_force'] = active_force
        traj_dict['conservative_force'] = conservative_force
        traj_dict['potential_energy'] = potential_energy
        traj_dict['image'] = image
        traj_dict['times'] = times
        traj_dict['edges'] = edges
        traj_dict['N'] = N
        traj_dict['dim'] = dim
        if has_topology:
            traj_dict['bonds'] = bonds

    else:
        traj = gsd.hoomd.open(myfile)
        traj_dict = {}
        has_topology = 0

        N = traj[0].particles.position.shape[0]
        try:
            log = gsd.hoomd.read_log(myfile)
        except:
            log = read_single_particle_log(myfile)

        nframes = traj.__len__()
        pos = []
        potential_energy = []
        image = []
        times = []
        edges = traj[0].configuration.box[:3]
        dim = traj[0].configuration.dimensions

        if traj[0].bonds.N>0:
            has_topology = 1
            bonds = []

        for n in range(nframes):
            frame = traj[n]
            pos.append(frame.particles.position)
            if 'log/particles/md/pair/LJ/energies' in log:
                potential_energy.append(np.sum(log['log/particles/md/pair/LJ/energies'][n,:]))
            elif 'log/particles/md/bond/Harmonic/energies' in log:
                potential_energy.append(np.sum(log['log/particles/md/bond/Harmonic/energies'][n,:]))
            elif 'log/particles/md/bond/FENEWCA/energies' in log:
                potential_energy.append(np.sum(log['log/particles/md/bond/FENEWCA/energies'][n,:]))
            else:
                potential_energy.append(0.0)
            image.append(frame.particles.image)
            if has_topology==1:
                bonds.append(frame.bonds.group)


        pos = np.array(pos)
        #print(pos)
        #print('max position:', np.max(pos))
        potential_energy = np.array(potential_energy)
        image = np.array(image)
        if has_topology==1:
            bonds = np.array(bonds)
        if 'log/particles/md/pair/LJ/forces' in log:
            conservative_force = log['log/particles/md/pair/LJ/forces']
            virial = log['log/particles/md/pair/LJ/virials']
        elif 'log/particles/md/bond/Harmonic/forces' in log:
            conservative_force = log['log/particles/md/bond/Harmonic/forces']
            virial = log['log/particles/md/bond/Harmonic/virials']
        elif 'log/particles/md/bond/FENEWCA/forces' in log:
            conservative_force = log['log/particles/md/bond/FENEWCA/forces']
            virial = log['log/particles/md/bond/FENEWCA/virials']
        else:
            conservative_force = np.zeros((nframes, pos.shape[1], pos.shape[2]))
            virial = np.zeros((nframes, pos.shape[1], 6))
        if 'log/particles/ActiveNoiseHoomd/ActiveNoiseForce/ActiveNoiseForce/forces' in log:
            active_force = log['log/particles/ActiveNoiseHoomd/ActiveNoiseForce/ActiveNoiseForce/forces']
        elif 'log/particles/ActiveNoiseForce/ActiveNoiseForce/forces' in log:
            #backwards compatibility
            active_force = log['log/particles/ActiveNoiseForce/ActiveNoiseForce/forces']
        else:
            active_force = np.zeros((nframes, pos.shape[1], pos.shape[2]))
        times = log['log/Time/time']
        vel = conservative_force + active_force #WARNING: assumes friction = 1!
        N = pos.shape[1]

        traj_dict['pos'] = pos
        traj_dict['vel'] = vel
        traj_dict['active_force'] = active_force
        traj_dict['conservative_force'] = conservative_force
        traj_dict['virial'] = virial
        traj_dict['potential_energy'] = potential_energy
        traj_dict['image'] = image
        traj_dict['times'] = times
        traj_dict['edges'] = edges
        traj_dict['N'] = N
        traj_dict['dim'] = dim
        if has_topology:
            traj_dict['bonds'] = bonds

    return traj_dict

def load_noise_traj(myfile):
    
    #Check whether this is an h5 file
    if not(myfile.endswith('.h5')):
        print('Error: input to "load_noise_traj" must be h5 file.')
        return {}

    traj = h5py.File(myfile)
    traj_dict = {}

    #noise parameters
    Lambda = np.array(traj['/parameters/Lambda'])
    tau = np.array(traj['/parameters/tau'])
    va = np.array(traj['/parameters/va'])

    #grid dimensions
    ncells = np.array(traj['/grid/dimensions'])
    spacing = np.array(traj['/grid/spacing'])
    dim = ncells.shape[0]

    #noise data
    dt = np.array(traj['/parameters/dt'])
    times = np.array(traj['/noise/timestep'])*dt
    
    if dim==1:
        noise = np.zeros((times.shape[0],ncells[0],1))
        noise = np.array(traj['/noise/value/x'])
    elif dim==2:
        noise = np.zeros((times.shape[0],ncells[0],ncells[1],2))
        noise[:,:,:,0] = np.array(traj['noise/value/x'])
        noise[:,:,:,1] = np.array(traj['noise/value/y'])
    elif dim==3:
        noise = np.zeros((times.shape[0],ncells[0],ncells[1],ncells[2],3))
        noise[:,:,:,:,0] = np.array(traj['noise/value/x'])
        noise[:,:,:,:,1] = np.array(traj['noise/value/y'])
        noise[:,:,:,:,2] = np.array(traj['noise/value/z'])
    else:
        print('Error: dim is ', dim, ', not 1, 2, or 3.')
        traj.close()
        exit()

    traj.close()

    #Assign data to dictionary
    traj_dict['tau'] = tau
    traj_dict['lambda'] = Lambda
    traj_dict['va'] = va

    traj_dict['dim'] = dim
    traj_dict['ncells'] = ncells
    traj_dict['spacing'] = spacing

    traj_dict['times'] = times
    traj_dict['noise'] = noise

    return traj_dict

def read_single_particle_log(name):

    """
    Copied from read_log 
    """
    if gsd is None:
        msg = 'gsd module is not available'
        raise RuntimeError(msg)

    with gsd.fl.open(
        name=str(name),
        mode='r',
        application='gsd.hoomd ' + gsd.version.version,
        schema='hoomd',
        schema_version=[1, 4],
    ) as gsdfileobj:
        logged_data_names = gsdfileobj.find_matching_chunk_names('log/')
        # Always log timestep associated with each log entry
        logged_data_names.insert(0, 'configuration/step')
        if len(logged_data_names) == 1:
            warnings.warn(
                'No logged data in file: ' + str(name), RuntimeWarning, stacklevel=2
            )

        logged_data_dict = dict()
        for log in logged_data_names:
            log_exists_frame_0 = gsdfileobj.chunk_exists(frame=0, name=log)
            is_configuration_step = log == 'configuration/step'


            if log_exists_frame_0 or is_configuration_step:
                if is_configuration_step and not log_exists_frame_0:
                    # handle default configuration step on frame 0
                    tmp = numpy.array([0], dtype=numpy.uint64)
                else:
                    tmp = gsdfileobj.read_chunk(frame=0, name=log)

                if not tmp.shape[0] == 1:
                    continue
                if tmp.shape[0] == 1 and len(tmp.shape)<2:
                    logged_data_dict[log] = np.full(
                        fill_value=tmp[0], shape=(gsdfileobj.nframes,)
                    )
                else:
                    logged_data_dict[log] = np.tile(
                        tmp, (gsdfileobj.nframes, *tuple(1 for _ in tmp.shape))
                    )

            for idx in range(1, gsdfileobj.nframes):
                for key in logged_data_dict.keys():
                    if not gsdfileobj.chunk_exists(frame=idx, name=key):
                        continue
                    data = gsdfileobj.read_chunk(frame=idx, name=key)
                    if len(logged_data_dict[key][idx].shape) == 0:
                        logged_data_dict[key][idx] = data[0]
                    else:
                        logged_data_dict[key][idx] = data

    return logged_data_dict

if __name__ == '__main__':
    myfile = sys.argv[1]
    traj = load_traj(myfile)
    print(traj)
