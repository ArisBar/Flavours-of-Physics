# coding=utf-8
import numpy as np
# new physical features 
def new_features(df):
    # significance of flight distance
    df['Flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    # momenta p0_pz (along the beam) of the muons
    df['p0_pz'] = np.sqrt(df['p0_p']**2 - df['p0_pt']**2)
    df['p1_pz'] = np.sqrt(df['p1_p']**2 - df['p1_pt']**2)
    df['p2_pz'] = np.sqrt(df['p2_p']**2 - df['p2_pt']**2)
    # momentum p_z (along the beam) of tau
    df['tau_pz'] = df['p0_pt']*np.sinh(df['p0_eta']) +\
                   df['p1_pt']*np.sinh(df['p1_eta']) +\
                   df['p1_pt']*np.sinh(df['p1_eta'])
    # modulus momentum p of tau
    df['tau_p'] = np.sqrt(df['pt']**2 + df['tau_pz']**2)
    # speed of tau
    df['tau_v'] = df['FlightDistance']/df['LifeTime']
    # mass of tau (momentum over speed)
    df['tau_m'] = df['tau_p']/df['tau_v']

    mmu = 105.6583715
    df['tau_e'] = np.sqrt(df.p0_p**2 + mmu**2) +\
                  np.sqrt(df.p1_p**2 + mmu**2) +\
                  np.sqrt(df.p2_p**2 + mmu**2)
    df['tau_m2'] = df.tau_e**2 - df.tau_p**2
    df['tau_m2'][df.tau_m2 < 0] = 0
    df['tau_m_k'] = np.sqrt(df.tau_m2)
    df['tau_m_t'] = df.tau_m * 299.792458
    return df


