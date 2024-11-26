import pytest
import netket as nk
import math
import numpy as np

from VMC import lattice
def test_lattice():
    """Testing the number of edges and nodes in the graphs, with PBC, created by NetKet match mathematical expectation."""
    L = [i for i in range(2,10)]
    for i in range(len(L)):
        graph = lattice(L[i])
        assert len([node for node in graph.nodes()]) == L[i]**2
        if L[i] == 2:
            assert  len([edge for edge in graph.edges()]) == 2*L[i]**2 -2*L[i]
        else:
            assert len([edge for edge in graph.edges()]) == 2*L[i]**2

from VMC import hammiltonian
from VMC import sx
from VMC import sy
from VMC import sz

def test_hamiltonain_implementation():
    """Test case for the creation of a Transverse field Ising hamiltonian on a 2D lattice with PBC taken from the VMC NetKet tutorial."""
    l_list=[i for i in range(2,5)] #Any larger values of l and NetKet will error from a memory error because this would be too large for a local operator!
    for l in l_list:
        hi = nk.hilbert.Spin(s=1/2, N=l**2)
        graph = lattice(l)
        hammiltonian_correct = nk.operator.Ising(hi, graph,  h=1, J=1)
        assert np.sum(np.abs(hammiltonian_correct.to_sparse() - hammiltonian(hi,l,1,1).to_sparse())**2) < 1e-5


from VMC import exact_diagonalisation
def test_exact_diagonalisation():
    """Tests the exact_diagonalisation function using a test case from the VMC NetKet tutorial."""
    hi = nk.hilbert.Spin(s=1/2, N=16)
    H = hammiltonian(hi,4,1,1)
    energy, state = exact_diagonalisation(H)

    assert energy.shape == ()
    assert state.shape == (hi.n_states, )
    assert -34.01060 < energy < -34.01059 #failing a factor of -1 has gone missing, energy = 34.010597.... right now



from VMC import MF
from VMC import initialise_params_MF
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

def test_model_MF():
    hi = nk.hilbert.Spin(s=1/2, N=16)
    inputs = hi.random_state(jax.random.key(1), (4,)) #Inputs to our model are random states of the hilbert space
    
    model, parmaters = initialise_params_MF(hi,True)
    
    log_psi = model.apply(parmaters, inputs)
    assert str(log_psi.shape) == "(4,)"

from VMC import to_array
def test_to_array():
    """Testing the to_array function using tests taken from the VMC from scratch NetKet tutorial."""
    hi = nk.hilbert.Spin(s=1/2, N=16)
    model, params = initialise_params_MF(hi,True)
    assert to_array(hi,model,params).shape == (hi.n_states, ) #assert that the dimension of the wavefunction is the same as the nubmer of configurations in the hilbert space
    assert np.all(to_array(hi,model,params)>0) #assert that all probabilities are >0
    np.testing.assert_allclose(np.linalg.norm(to_array(hi,model,params)), 1.0) #no idea what this test is

from VMC import compute_energy
def test_compute_energy():
    hi = nk.hilbert.Spin(s=1/2, N=16)
    H = hammiltonian(hi,4,1,1).to_sparse()
    model, params = initialise_params_MF(hi,True)
    assert compute_energy(hi,model,params,H).shape == ()
    assert compute_energy(hi,model,params,H) < 0


from VMC import Jastrow
from VMC import initialise_params_Jastrow

def test_model_Jastrow():
    hi = nk.hilbert.Spin(s=1/2, N=9)
    model_jastrow = Jastrow()

    one_sample = hi.random_state(jax.random.key(0)) #One sample
    batch_samples = hi.random_state(jax.random.key(0), (5,)) #5 samples
    multibatch_samples = hi.random_state(jax.random.key(0), (5,4)) #4 batches of 5 samples

    parameters_jastrow = model_jastrow.init(jax.random.key(0), one_sample)

    assert parameters_jastrow['params']['J'].shape == (hi.size, hi.size) #assert the Jastrow matrix is of the correct dimensionality
    assert model_jastrow.apply(parameters_jastrow, one_sample).shape == () #The result of the einstein summation on the one sample should be a scaler
    assert model_jastrow.apply(parameters_jastrow, batch_samples).shape == batch_samples.shape[:-1] #The result of the batch of einstien summations should be a batch of results
    assert model_jastrow.apply(parameters_jastrow, multibatch_samples).shape == multibatch_samples.shape[:-1]

from VMC import compute_local_energies
from VMC import sigma_eta_identification
from VMC import MC_initialise_samples

def test_local_energies():

    hi = nk.hilbert.Spin(s=1/2, N=9)
    model, params = initialise_params_MF(hi, True)
    H = hammiltonian(hi,3,1,1).to_jax_operator()
    samples, sampler, sampler_state = MC_initialise_samples(hi,100,model, params)
    #samples, sampler_state = sampler.sample(model,params, state = sampler_state,chain_length=1)

    assert compute_local_energies(model,params,H, samples[0]).shape == samples.shape[1:-1]
    assert compute_local_energies(model,params,H, samples).shape == samples.shape[:-1]

from VMC import estimate_energy
def test_estimate_energy():
    hi = nk.hilbert.Spin(s=1/2, N=9)
    model, params = initialise_params_MF(hi, True)
    H = hammiltonian(hi,3,1,1).to_jax_operator()
    samples, sampler, sampler_state = MC_initialise_samples(hi,100,model, params)
    E_loc = compute_local_energies(model,params,H,samples)

    isinstance(estimate_energy(model,params,H,samples), nk.stats.Stats)
    print(estimate_energy(model,params,H,samples))
