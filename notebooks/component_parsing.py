"""
Parsing utilities for component state names in structural time series models.

This module provides functionality to parse complex state names like 'trend[level[observed_state]]'
into structured multi-index coordinates that enable easy component and state selection.

NB: This is still a work in progress, and probably need to be expanded to more complex cases.
"""

from __future__ import annotations

import logging
import re
from functools import partial

from collections.abc import Sequence

import pandas as pd
import xarray as xr

# Set up logging
_log = logging.getLogger(__name__)

def extract_components_from_idata(
        self, idata: xr.Dataset, restructure: bool = False
    ) -> xr.Dataset:
        r"""
        Extract interpretable hidden states from an InferenceData returned by a PyMCStateSpace sampling method

        Parameters
        ----------
        idata : Dataset
            A Dataset object, returned by a PyMCStateSpace sampling method
        restructure : bool, default False
            Whether to restructure the state coordinates as a multi-index for easier component selection.
            When True, enables selections like `idata.sel(component='level')` and `idata.sel(observed='gdp')`.
            Particularly useful for multivariate models with multiple observed states.

        Returns
        -------
        idata : Dataset
            A Dataset object with hidden states transformed to represent only the "interpretable" subcomponents
            of the structural model. If `restructure=True`, the state coordinate will be a multi-index with
            levels ['component', 'observed'] for easier selection.

        Notes
        -----
        In general, a structural statespace model can be represented as:

        .. math::
            y_t = \mu_t + \nu_t + \cdots + \gamma_t + c_t + \xi_t + \epsilon_t \tag{1}

        Where:

            - :math:`\mu_t` is the level of the data at time t
            - :math:`\nu_t` is the slope of the data at time t
            - :math:`\cdots` are higher time derivatives of the position (acceleration, jerk, etc) at time t
            - :math:`\gamma_t` is the seasonal component at time t
            - :math:`c_t` is the cycle component at time t
            - :math:`\xi_t` is the autoregressive error at time t
            - :math:`\varepsilon_t` is the measurement error at time t

        In state space form, some or all of these components are represented as linear combinations of other
        subcomponents, making interpretation of the outputs difficult. The purpose of this function is
        to take the expended statespace representation and return a "reduced form" of only the components shown in
        equation (1).

        When `restructure=True`, the returned dataset allows for easy component selection, especially for
        multivariate models with multiple observed states.
        """

        def _extract_and_transform_variable(idata, new_state_names):
            *_, time_dim, state_dim = idata.dims
            state_func = partial(self._hidden_states_from_data)
            new_idata = xr.apply_ufunc(
                state_func,
                idata,
                input_core_dims=[[time_dim, state_dim]],
                output_core_dims=[[time_dim, state_dim]],
                exclude_dims={state_dim},
            )
            new_idata.coords.update({state_dim: new_state_names})
            return new_idata

        var_names = list(idata.data_vars.keys())
        is_latent = [idata[name].shape[-1] == self.k_states for name in var_names]
        new_state_names = self._get_subcomponent_names()

        latent_names = [name for latent, name in zip(is_latent, var_names) if latent]
        dropped_vars = set(var_names) - set(latent_names)
        if len(dropped_vars) > 0:
            _log.warning(
                f"Variables {', '.join(dropped_vars)} do not contain all hidden states (their last dimension "
                f"is not {self.k_states}). They will not be present in the modified idata."
            )
        if len(dropped_vars) == len(var_names):
            raise ValueError(
                "Provided idata had no variables with all hidden states; cannot extract components."
            )

        idata_new = xr.Dataset(
            {
                name: _extract_and_transform_variable(idata[name], new_state_names)
                for name in latent_names
            }
        )

        if restructure:
            try:
                idata_new = restructure_components_idata(idata_new)
            except Exception as e:
                _log.warning(
                    f"Failed to restructure components with multi-index: {e}. "
                    "Returning dataset with original string-based state names. "
                    "You can call restructure_components_idata() manually if needed."
                )

        return idata_new

def parse_component_state_name(state_name: str) -> tuple[str, str]:
    """
    Parse a component state name into its constituent parts.

    Extracts the actual interpretable state name and observed state from
    various component naming formats.

    Parameters
    ----------
    state_name : str
        The state name to parse, e.g., 'trend[level[observed_state]]' or 'ar[observed_state]'

    Returns
    -------
    tuple[str, str]
        A tuple of (component, observed) where component is the interpretable component name
        and observed is the observed state name

    Examples
    --------
    >>> parse_component_state_name('trend[level[chirac2]]')
    ('level', 'chirac2')
    >>> parse_component_state_name('ar[macron]')
    ('ar', 'macron')
    """
    # Handle the nested bracket pattern: component[state[observed]]
    # For these, we want the inner state name (level, trend, etc.)
    # because the first level is redundant with the component name
    nested_pattern = r"^([^[]+)\[([^[]+)\[([^]]+)\]\]$"
    nested_match = re.match(nested_pattern, state_name)

    if nested_match:
        # Return the inner state name and observed state
        return nested_match.group(2), nested_match.group(3)

    # Handle the simple bracket pattern: component[observed]
    # For these, we want the component name directly
    simple_pattern = r"^([^[]+)\[([^]]+)\]$"
    simple_match = re.match(simple_pattern, state_name)

    if simple_match:
        # Return the component name and observed state
        return simple_match.group(1), simple_match.group(2)

    # If no pattern matches, treat the whole string as a state name
    # This is a fallback for edge cases
    return state_name, "default"


def create_component_multiindex(
    state_names: Sequence[str], coord_name: str = "state"
) -> xr.Coordinates:
    """
    Create xarray coordinates with multi-index from component state names.

    Parameters
    ----------
    state_names : Sequence[str]
        List of state names to parse into multi-index
    coord_name : str, default "state"
        Name for the coordinate dimension to transform into a multi-index

    Returns
    -------
    xr.Coordinates
        xarray coordinates with multi-index structure

    Examples
    --------
    >>> state_names = ['trend[level[observed_state]]', 'trend[trend[observed_state]]', 'ar[observed_state]']
    >>> coords = create_component_multiindex(state_names)
    >>> coords.to_index().names
    ['component', 'observed']
    >>> coords.to_index().values
    [('level', 'observed_state'), ('trend', 'observed_state'), ('ar', 'observed_state')]
    """
    tuples = [parse_component_state_name(name) for name in state_names]
    midx = pd.MultiIndex.from_tuples(tuples, names=["component", "observed"])

    return xr.Coordinates.from_pandas_multiindex(midx, dim=coord_name)


def restructure_components_idata(idata: xr.Dataset) -> xr.Dataset:
    """
    Restructure idata with multi-index coordinates for easier component selection.

    Parameters
    ----------
    idata : xr.Dataset
        Dataset with component state names as coordinates

    Returns
    -------
    xr.Dataset
        Dataset with restructured multi-index coordinates

    Examples
    --------
    >>> # After calling extract_components_from_idata from core.py
    >>> restructured = restructure_components_idata(components_idata)
    >>> # Now you can select by component or observed state
    >>> level_data = restructured.sel(component='level')  # All level components
    >>> gdp_data = restructured.sel(observed='gdp')  # All gdp data
    >>> level_gdp = restructured.sel(component='level', observed='gdp')  # Specific combination
    """
    # name of the coordinate containing state names
    # should be `state`, by default, as users don't access it directly
    # would need to be updated if we want to support custom names
    state_coord_name = "state"
    if state_coord_name not in idata.coords:
        raise ValueError(f"Coordinate '{state_coord_name}' not found in dataset")

    state_names = idata.coords[state_coord_name].values
    mindex_coords = create_component_multiindex(state_names, state_coord_name)

    return idata.assign_coords(mindex_coords)
