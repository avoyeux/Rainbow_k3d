"""
To store dataclasses related only to the warping of the data.
"""

from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field

# TYPE ANNOTATIONs
from typing import Self, Literal, cast

# API public
__all__ = [
    'WarpedInformation',
    'WarpedIntegration',
    'WarpedDataGroup',
    'AllWarpedInformation',
]

# todo take away the __radd__ method and the __add__ method from the WarpedInformation class



@dataclass(slots=True, repr=False, eq=False)
class WarpedInformation:
    """
    To store all the warped information data for each date.
    It also stores the corresponding polynomial fit angles.
    """

    # todo add the contours also in the integration ?

    # DATA
    warped_values: np.ndarray
    angles: np.ndarray

    # METADATA
    name: str
    date: str
    fit_order: int
    arc_length: np.ndarray  # in km
    integration_time: int | None
    integration_type: Literal['mean', 'median'] = 'mean'

    def warped_integration(self) -> None:
        """
        To compute the integration of the warped values.
        The values are saved back into the warped_values attribute.

        Raises:
            ValueError: if the integration type is not 'mean' or 'median'.
        """

        if self.integration_type == 'mean':
            self.warped_values = np.mean(self.warped_values, axis=0)
        elif self.integration_type == 'median':
            self.warped_values = np.median(self.warped_values, axis=0)
        else:
            raise ValueError(
                f"\033[1;31mUnknown integration type: {self.integration_type}. "
                "Choose between 'mean' and 'median'.\033[0m"
            )
    
    def check_adding(self, other: Self | WarpedIntegration) -> bool:
        """
        To check if the two objects can be added together.

        Args:
            other (Self | WarpedIntegration): the two instances to check.

        Returns:
            bool: True if the two instances can be added together, False otherwise.
        """

        check_fit_order = (self.fit_order == other.fit_order)
        check_integration_type = (self.integration_type == other.integration_type)
        return check_integration_type & check_fit_order


@dataclass(slots=True, repr=False, eq=False)
class WarpedIntegration:
    """
    To store the warped information for all the dates for a given dataset, fit order and
    integration time.
    """

    # ! make sure that the cadence is the same for all the dates or do a time interpolation later

    # METADATA
    name: str
    dates: list[str]
    fit_order: int
    integration_type: Literal['mean', 'median']

    # DATA
    arc_lengths: list[np.ndarray]
    warped_informations: list[WarpedInformation]  # * could change it to a sequence for subclassing

    # METADATA optional
    integration_time: int | None

    def sort(self) -> None:
        """
        To sort the different list attributes of the class instance by date.
        """

        # INDICEs sorting
        sorting_indices = np.argsort(self.dates)

        # ATTRIBUTEs sorting
        self.dates[:] = [self.dates[i] for i in sorting_indices]
        self.arc_lengths[:] = [self.arc_lengths[i] for i in sorting_indices]
        self.warped_informations[:] = [self.warped_informations[i] for i in sorting_indices]

    def append(self, other: WarpedInformation) -> None:
        """
        To append a new warped information to the current one.
        If the warped information to append has the same metadata than a current one, then a
        TypeError is raised. This is to avoid duplicates and badly defined data.

        Args:
            other (WarpedInformation): the other instance to append to the current one.

        Raises:
            TypeError: if the other instance is not a WarpedInformation or if the metadata is the
                same as a current value in the warped_informations instance list.
        """
        
        # CHECK append
        if not self._append_checks(other):
            raise TypeError(
                f"\033[1;31mCannot append {type(other)} to {type(self)}. "
                "Could be due to wrong type or duplicate warped information instances.\033[0m"
            )

        # METADATA update
        self.dates.append(other.date)

        # DATA update
        self.warped_informations.append(other)
        self.arc_lengths.append(other.arc_length)

    def _append_checks(self, other: WarpedInformation) -> bool:
        """
        To check if the other instance can be safely (without duplicates) added to the current
        instance.

        Args:
            other (WarpedInformation): the other instance to check.

        Returns:
            bool: True if the other instance can be added to the current instance, False otherwise.
        """

        # CHECK type
        if not isinstance(other, WarpedInformation): return False 

        # CHECK attributes
        check_date = not (other.date in self.dates)
        check_fit_order = (self.fit_order == other.fit_order)
        check_integration_type = (self.integration_type == other.integration_type)
        check_integration_time = (self.integration_time == other.integration_time)
        return any([check_date, check_fit_order, check_integration_type, check_integration_time])


@dataclass(slots=True, repr=False, eq=False)
class WarpedDataGroup:
    """
    To store the warped information for a given dataset.
    In this case the default value for most of the attributes is an empty list (to make appending
    values to the class instance easier).
    """

    # METADATA
    name: str
    fit_orders: list[int] = field(default_factory=list)
    integration_times: list[int | None] = field(default_factory=list)

    # DATA
    warped_integrations: list[WarpedIntegration] = field(default_factory=list)

    def append(self, other: WarpedInformation | WarpedIntegration) -> None:
        """
        To append a new warped integration or warped information to the current instance.
        If the integration time or fit order is not already in the list, it is added to the list.

        Args:
            other (WarpedInformation | WarpedIntegration): the warped information or integration to
                append.

        Raises:
            TypeError: if the other instance is not a WarpedInformation or WarpedIntegration.
        """

        # APPEND basic
        if isinstance(other, WarpedIntegration):  # ! this might create problems if duplicates
            # METADATA update
            if other.fit_order not in self.fit_orders: self.fit_orders.append(other.fit_order)
            if other.integration_time not in self.integration_times:
                self.integration_times.append(other.integration_time)

            # DATA update
            self.warped_integrations.append(other)
        # APPEND inside
        elif isinstance(other, WarpedInformation):
            self._append_sub_attributes(other)
        # WRONG TYPE
        else:
            raise TypeError(
                f"\033[1;31mCannot append {type(other)} to {type(self)}.\033[0m"
            )

    def _append_sub_attributes(self, other: WarpedInformation) -> None:
        """
        To append data directly to the warped integrations inside the warped_integration list.
        If there is no warped integration corresponding to the warped information data, a new
        warped integration dataclass instance is added to the warped_integration instance
        attribute.

        Args:
            other (WarpedInformation): the warped information to append to a warped integration
                instance.
        """
        
        # INTEGRATION new
        if any([
            other.fit_order not in self.fit_orders,
            other.integration_time not in self.integration_times,
            ]):

            # METADATA update
            if other.fit_order not in self.fit_orders: self.fit_orders.append(other.fit_order)
            if other.integration_time not in self.integration_times:
                self.integration_times.append(other.integration_time)

            # DATA update
            self.warped_integrations.append(WarpedIntegration(
                name=f"warped integration",
                dates=[other.date],
                integration_time=other.integration_time,
                arc_lengths=[other.arc_length],
                fit_order=other.fit_order,
                integration_type=other.integration_type,
                warped_informations=[other],
            ))
        # INTEGRATION update
        else:
            for integration in self.warped_integrations:
                # FIND correct integration
                if all([  # ! cannot have multiple integration types
                    integration.fit_order == other.fit_order,
                    integration.integration_type == other.integration_type,
                    ]):

                    # DATA update
                    integration.append(other)
                    break


@dataclass(slots=True, repr=False, eq=False)
class AllWarpedInformation:
    """
    To store all the warped information of all the dates for all datasets.
    """

    # DATASETs
    integration: WarpedDataGroup | None = None
    full_integration_no_duplicates: WarpedDataGroup | None = None
