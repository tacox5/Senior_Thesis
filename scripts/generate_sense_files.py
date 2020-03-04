import numpy as np
import pickle
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum


ps = np.load('ps_21.npz')
k = ps['k']
zs = ps['zs']
nu_s = ps['nu_s']

lat = -30.72 * np.pi / 180.
layout_350 = np.loadtxt('../hera_positions_staged/antenna_positions_350.dat',
                        delimiter=' ')

def main(i):
    print (nu_s[i], zs[i])
    sensitivity = PowerSpectrum(
            observation = Observation(
                observatory = Observatory(
                    antpos = layout_350,
                    beam = GaussianBeam(frequency=nu_s[i], dish_size=14),
                    latitude = lat
                )
            ),
            foreground_model = 'moderate',
            k_21 = k,
            delta_21 = ps[str(zs[i])]
        )
    pkl_name = 'hera_sense_mod_{}'.format(str(zs[i]).replace('.', '_'))

    with open(pkl_name + '.pkl', 'wb') as pfile:
        pickle.dump(sensitivity, pfile)

if __name__ == '__main__':
    for i in range(len(zs)):
        main(i)
