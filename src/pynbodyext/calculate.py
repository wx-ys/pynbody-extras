""" 


"""

from abc import ABC, abstractmethod
from typing import TypeVar,Generic, Union

from pynbody import units
from pynbody.snapshot import SimSnap

ReturnT = TypeVar("ReturnT",covariant=True)

class CalculatorBase(Generic[ReturnT], ABC):
    """An abstract base class for performing calculations on particle data.
    """
    
    @abstractmethod
    def __call__(self, sim: SimSnap) -> ReturnT:
        """Executes the calculation on a given simulation snapshot.

        This is the main public method to run the calculator.

        Parameters
        ----------
        sim : pynbody.snapshot.SimSnap
            The particle data (a snapshot or sub-snapshot) on which to perform
            the calculation.

        Returns
        -------
        ReturnT
            The result of the calculation. The exact type depends on the subclass.
        
        Raises
        ------
        NotImplementedError
            If a subclass does not override this method.
        """
        
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__(self, sim: SimSnap) -> ReturnT"
        )


    
    def _in_sim_units(
        self,
        value: Union[str, units.UnitBase, float, int],
        sim_parameter: str,
        sim: SimSnap,
    ) -> float:
        """
        Convert a value into the simulation's native units for a given array.

        This converts `value` - which may be a string with units (e.g., "10 kpc"),
        a pynbody Unit object, or a raw number assumed to already be in native units -
        to a float expressed in the units of `sim[sim_parameter]`.

        Parameters
        ----------
        value : str or pynbody.units.UnitBase or float or int
            The value to be converted.
            - str: parsed by `pynbody.units.Unit`, e.g., "10 kpc", "200 km s^-1".
            - UnitBase: converted using `.in_units(target_units, **context)`.
            - number: treated as already in the target native units.
        sim_parameter : str
            The name of the simulation array (e.g., 'r', 'rxy', 'mass') whose units
            define the conversion target.
        sim : pynbody.snapshot.SimSnap
            The simulation snapshot.

        Returns
        -------
        float
            The numeric value in the native units of `sim[sim_parameter]`.
        """
        if isinstance(value, str):
            value = units.Unit(value)
        if units.is_unit_like(value):
            value = float(value.in_units(sim[sim_parameter].units,
                                          **sim[sim_parameter].conversion_context()))
        return value
    
    
    def __repr__(self):
        if hasattr(self,"name"):
            return f"<Calculator {self.name}()>"
        return f"<Calculator {self.__class__.__name__}()>"