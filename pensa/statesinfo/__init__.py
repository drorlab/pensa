"""
Methods for state-specific information.

The methods here are based on the following paper:

    |    Neil J. Thomson, Owen N. Vickery, Callum M. Ives, Ulrich Zachariae:
    |    Ion-water coupling controls class A GPCR signal transduction pathways.
    |    https://doi.org/10.1101/2020.08.28.271510

"""
from .discrete_states import \
    smart_gauss_fit, \
    get_intersects, \
    determine_state_limits, \
    get_discrete_states, \
    calculate_entropy, \
    calculate_entropy_multthread
